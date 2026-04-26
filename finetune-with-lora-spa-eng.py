"""
LoRA fine-tuning on N hours of SwitchLingua code-switching audio data.

Reads transcripts directly from the SwitchLingua_audio CSV's `text` column.
Does NOT use SwitchLingua_text -- everything we need is in the audio repo.

Training only. Eval is handled by a separate script that consumes
`test_pool.json` (data NOT used for training, written to output_dir).

Supported language pairs:
    ara, yue, fra, deu, hin, ita, jpn, kor, zho, rus, spa  (each x English)
    mucs_hin_eng  (OpenSLR MUCS Hindi-English)

Usage:
    # First run -- downloads <Language>/ folder + <Language>.csv
    python finetune_lora_switchlingua.py --lang_pair fra --device cuda

    # Subsequent runs -- skips Hub round-trip
    python finetune_lora_switchlingua.py --lang_pair fra --device cuda --skip_download
"""
import os, sys, json, random, tarfile, urllib.request, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa, numpy as np, torch, soundfile as sf, pandas as pd
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(str(Path(__file__).parent / "src"))
try:
    from cs_synth_to_real.utils.text_norm import normalize_text  # noqa: F401
except ImportError:
    def normalize_text(s: str) -> str:
        return s.lower().strip()

import warnings
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa.*")
# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------
# code -> {audio_csv, audio_dir, expected_lang}
# expected_lang is the value we expect to find in the CSV's first-language
# column. If the CSV says something else, we abort.
SWITCHLINGUA_LANGUAGES: Dict[str, Dict[str, str]] = {
    "ara": {"audio_csv": "Arabic.csv",    "audio_dir": "Arabic",    "expected_lang": "Arabic"},
    "yue": {"audio_csv": "Cantonese.csv", "audio_dir": "Cantonese", "expected_lang": "Cantonese"},
    "fra": {"audio_csv": "French.csv",    "audio_dir": "French",    "expected_lang": "French"},
    "deu": {"audio_csv": "German.csv",    "audio_dir": "German",    "expected_lang": "German"},
    "hin": {"audio_csv": "Hindi.csv",     "audio_dir": "Hindi",     "expected_lang": "Hindi"},
    "ita": {"audio_csv": "Italian.csv",   "audio_dir": "Italian",   "expected_lang": "Italian"},
    "jpn": {"audio_csv": "Japanese.csv",  "audio_dir": "Japanese",  "expected_lang": "Japanese"},
    "kor": {"audio_csv": "Korean.csv",    "audio_dir": "Korean",    "expected_lang": "Korean"},
    "zho": {"audio_csv": "Mandarin.csv",  "audio_dir": "Mandarin",  "expected_lang": "Mandarin"},
    "rus": {"audio_csv": "Russian.csv",   "audio_dir": "Russian",   "expected_lang": "Russian"},
    "spa": {"audio_csv": "Spanish.csv",   "audio_dir": "Spanish",   "expected_lang": "Spanish"},
}
MUCS_LANG_PAIRS = {"mucs_hin_eng"}
ALL_LANG_PAIRS = list(SWITCHLINGUA_LANGUAGES.keys()) + list(MUCS_LANG_PAIRS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(t: str) -> str:
    return t.replace("**", "").replace("__", "").strip()


def build_joint_vocab(texts: List[str]) -> Dict[str, int]:
    chars = set()
    for t in texts:
        if t:
            chars.update(clean_text(t))
    chars.discard("\n"); chars.discard("\t")
    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    if " " in vocab:
        vocab["|"] = vocab.pop(" ")
    else:
        vocab["|"] = len(vocab)
    return vocab


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column, case- and space-insensitive."""
    norm = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_").replace("-", "_")
        if key in norm:
            return norm[key]
    return None


# ---------------------------------------------------------------------------
# MUCS loader
# ---------------------------------------------------------------------------

MUCS_HIN_ENG_URL = "https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz"
MUCS_HIN_ENG_FILENAME = "Hindi-English_train.tar.gz"


def _is_valid_tar(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as tf:
            for _ in tf:
                pass
        return True
    except Exception:
        return False


def download_mucs(data_dir: Path, url: str, filename: str, lang: str,
                  skip_download: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / filename
    extract_dir = data_dir / filename.replace(".tar.gz", "")

    if skip_download and extract_dir.exists():
        print(f"  [skip_download] Using existing extraction: {extract_dir}")
        return extract_dir

    def _hook(b, bs, tot):
        if tot > 0:
            print(f"\r  {min(b * bs / tot * 100, 100):.1f}%", end="", flush=True)

    for attempt in range(1, 4):
        if not tar_path.exists():
            print(f"  Downloading MUCS {lang} (attempt {attempt}/3)...")
            urllib.request.urlretrieve(url, tar_path, reporthook=_hook)
            print()
        if not _is_valid_tar(tar_path):
            tar_path.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Failed to download {lang} after 3 attempts.")
            continue
        if not extract_dir.exists():
            print(f"  Extracting {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(path=data_dir)
            print("  Done.")
        break
    return extract_dir


def load_mucs_entries(extract_dir: Path) -> List[Dict]:
    text_files = list(extract_dir.rglob("text"))
    if not text_files:
        raise FileNotFoundError(f"No text file under {extract_dir}")
    text_file = text_files[0]
    data_dir = text_file.parent

    utt2text: Dict[str, str] = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1]

    utt2seg: Dict[str, Dict] = {}
    seg_file = data_dir / "segments"
    if seg_file.exists():
        with open(seg_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    utt2seg[parts[0]] = {
                        "wav_id": parts[1],
                        "start": float(parts[2]),
                        "end": float(parts[3]),
                    }

    wav_id2path = {p.stem: p for p in extract_dir.rglob("*.wav")}
    print(f"  Found {len(wav_id2path)} wav files")

    entries, missing = [], 0
    for utt_id, transcript in utt2text.items():
        seg = utt2seg.get(utt_id)
        if seg:
            wav_path = wav_id2path.get(seg["wav_id"])
            if not wav_path:
                missing += 1
                continue
            start, end = seg["start"], seg["end"]
        else:
            wav_path = wav_id2path.get(utt_id)
            if not wav_path:
                missing += 1
                continue
            start, end = 0.0, None
        try:
            info = sf.info(str(wav_path))
            dur = (end or info.duration) - start
        except Exception:
            try:
                sig, sr = librosa.load(str(wav_path), sr=None)
                dur = ((end or len(sig) / sr) - start)
            except Exception as e:
                print(f"  Warning: {wav_path}: {e}")
                continue
        entries.append({
            "audio_path": str(wav_path),
            "text": transcript,
            "duration": dur,
            "start_time": start,
            "end_time": end or (start + dur),
            "utt_id": utt_id,
        })
    if missing:
        print(f"  {missing} transcripts skipped (no wav).")
    print(f"  Loaded {len(entries)} MUCS entries.")
    return entries


# ---------------------------------------------------------------------------
# SwitchLingua: download + load (audio repo only)
# ---------------------------------------------------------------------------

def download_switchlingua_audio(sl_dir: Path, lang_cfg: Dict[str, str],
                                 skip_download: bool = False) -> str:
    """Download just the audio repo for one language. Skips if already local."""
    audio_dir = sl_dir / "audio"
    audio_csv = audio_dir / lang_cfg["audio_csv"]
    audio_files_dir = audio_dir / lang_cfg["audio_dir"]

    have_audio = (
        audio_csv.exists() and audio_csv.stat().st_size > 0
        and audio_files_dir.exists() and any(audio_files_dir.iterdir())
    )

    if skip_download:
        if not have_audio:
            raise FileNotFoundError(
                f"--skip_download set but local data missing for {lang_cfg['expected_lang']}.\n"
                f"  Expected: {audio_csv} and {audio_files_dir}/\n"
                f"  Run once without --skip_download first."
            )
        print(f"  [skip_download] Using local audio data for {lang_cfg['expected_lang']}")
        return str(audio_dir)

    if have_audio:
        print(f"  Local audio data found for {lang_cfg['expected_lang']}. "
              f"Verifying with Hub (use --skip_download to bypass)...")

    from huggingface_hub import snapshot_download
    hf_token = os.environ.get("HF_TOKEN", True)

    try:
        print(f"  Resolving SwitchLingua_audio for {lang_cfg['expected_lang']}...")
        snapshot_download(
            "Shelton1013/SwitchLingua_audio",
            repo_type="dataset",
            local_dir=str(audio_dir),
            allow_patterns=[f"{lang_cfg['audio_dir']}/*", lang_cfg["audio_csv"]],
            token=hf_token,
        )
    except Exception as e:
        err_str = str(e)
        err_type = type(e).__name__
        if "LocalTokenNotFound" in err_type:
            raise SystemExit(
                "\n[ERROR] No HuggingFace token found.\n"
                "  Run `hf auth login` (or `huggingface-cli login`) first,\n"
                "  or export HF_TOKEN.\n"
            ) from e
        if "401" in err_str or "GatedRepo" in err_type or "Unauthorized" in err_str or "403" in err_str:
            raise SystemExit(
                "\n[ERROR] Cannot access Shelton1013/SwitchLingua_audio.\n"
                "  1. Visit https://huggingface.co/datasets/Shelton1013/SwitchLingua_audio while logged in\n"
                "  2. Click 'Agree and access repository'\n"
                "  3. Per the dataset card, also email signed agreement to Shelton1013@outlook\n"
                "  4. Wait for approval, then re-run.\n"
                f"  Underlying error: {err_type}: {err_str[:200]}"
            ) from e
        raise

    return str(audio_dir)


def load_switchlingua_entries(audio_snapshot: str,
                               lang_cfg: Dict[str, str]) -> List[Dict]:
    """Load entries from the audio repo CSV. No text repo needed."""
    expected_lang = lang_cfg["expected_lang"]
    audio_csv_path = os.path.join(audio_snapshot, lang_cfg["audio_csv"])
    if not os.path.exists(audio_csv_path):
        available = sorted(p.name for p in Path(audio_snapshot).glob("*.csv"))
        raise FileNotFoundError(
            f"Audio CSV {audio_csv_path} not found.\n"
            f"  Available: {available}"
        )

    df = pd.read_csv(audio_csv_path, low_memory=False)
    print(f"  Audio CSV: {audio_csv_path} ({len(df)} rows)")
    print(f"  Columns: {list(df.columns)}")

    # ---- Locate columns flexibly (CSVs across languages have inconsistent
    # naming: "first_language" vs "first Language", etc.) ----
    audio_col = find_column(df, ["file_name", "filename", "audio_file", "path"])
    text_col = find_column(df, ["text", "transcript", "sentence"])
    first_lang_col = find_column(df, ["first_language", "first Language", "L1"])
    second_lang_col = find_column(df, ["second_language", "second Language", "L2"])
    conv_type_col = find_column(df, ["conversation_type"])
    speaker_col = find_column(df, ["speaker_id", "speaker", "spk_id"])

    if audio_col is None or text_col is None:
        raise RuntimeError(
            f"Required columns missing.\n"
            f"  Need at least one of: file_name/filename/audio_file (got {audio_col})\n"
            f"  And: text/transcript/sentence (got {text_col})\n"
            f"  Available columns: {list(df.columns)}"
        )

    print(f"  -> audio: {audio_col!r}, text: {text_col!r}, "
          f"L1: {first_lang_col!r}, L2: {second_lang_col!r}, "
          f"conv: {conv_type_col!r}, speaker: {speaker_col!r}")

    # ---- Sanity check: does this CSV actually contain the requested language? ----
    if first_lang_col or second_lang_col:
        first_langs = (set(df[first_lang_col].astype(str).str.strip().unique())
                       if first_lang_col else set())
        second_langs = (set(df[second_lang_col].astype(str).str.strip().unique())
                        if second_lang_col else set())
        all_langs = first_langs | second_langs
        print(f"  Languages in CSV: L1={first_langs}, L2={second_langs}")

        if expected_lang not in all_langs:
            raise RuntimeError(
                f"\n[ERROR] {audio_csv_path} does not contain {expected_lang} data!\n"
                f"  L1 values: {first_langs}\n"
                f"  L2 values: {second_langs}\n"
                f"  Requested: {expected_lang}\n"
                f"  This is dataset-side mislabeling. Try a different --lang_pair.\n"
            )

        # If the CSV is mixed, filter to rows containing expected_lang
        contaminants = all_langs - {expected_lang, "English"}
        if contaminants:
            before = len(df)
            mask = pd.Series(False, index=df.index)
            if first_lang_col:
                mask |= (df[first_lang_col].astype(str).str.strip() == expected_lang)
            if second_lang_col:
                mask |= (df[second_lang_col].astype(str).str.strip() == expected_lang)
            df = df[mask].reset_index(drop=True)
            print(f"  Filtered to {expected_lang}-containing rows: {before} -> {len(df)} "
                  f"(removed contaminants: {contaminants})")
            if len(df) == 0:
                raise RuntimeError(f"No {expected_lang} rows after filtering.")
    else:
        print(f"  WARNING: no language columns found. Cannot verify content matches "
              f"{expected_lang}. Proceeding anyway.")

    # ---- Optional: filter to single-turn utterances ----
    # Note: with text from the audio CSV's per-utterance `text` column, this
    # filter is less critical -- multi-turn rows still have one transcript
    # per audio file. Disabled by default; flip if you want it stricter.
    if conv_type_col is not None:
        ct = (df[conv_type_col].astype(str).str.lower().str.strip()
              .str.replace("_", " ").str.replace("-", " "))
        single_turn_count = int((ct == "single turn").sum())
        multi_turn_count = int((ct == "multi turn").sum())
        print(f"  conversation_type: single_turn={single_turn_count}, "
              f"multi_turn={multi_turn_count}")

    # ---- Build entries ----
    audio_base = os.path.join(audio_snapshot, lang_cfg["audio_dir"])
    entries, missing_audio, bad_text, bad_audio = [], 0, 0, 0

    for i, row in df.iterrows():
        rel_audio = str(row[audio_col]).strip()
        transcript = str(row[text_col]).strip()

        if not transcript or transcript.lower() in ("nan", "none", ""):
            bad_text += 1
            continue

        audio_path = os.path.join(audio_base, rel_audio)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(audio_snapshot, rel_audio)
        if not os.path.exists(audio_path):
            missing_audio += 1
            continue

        # Determine duration. soundfile only handles PCM/FLAC/OGG; for m4a/mp3
        # use librosa (which goes through audioread -> ffmpeg). We pick the
        # path by extension to avoid 1000s of "PySoundFile failed" warnings.
        ext = os.path.splitext(audio_path)[1].lower()
        sf_compatible = ext in (".wav", ".flac", ".ogg", ".aiff", ".aif")
        dur = None
        if sf_compatible:
            try:
                info = sf.info(audio_path)
                dur = info.duration
            except Exception:
                pass
        if dur is None:
            try:
                sig, sr_local = librosa.load(audio_path, sr=None)
                dur = len(sig) / sr_local
            except Exception as e:
                if bad_audio < 3:
                    print(f"  Cannot read {audio_path}: {type(e).__name__}: {e}")
                bad_audio += 1
                continue

        entry = {
            "audio_path": audio_path,
            "text": transcript,
            "duration": dur,
            "start_time": 0.0,
            "end_time": dur,
        }
        if speaker_col and pd.notna(row[speaker_col]):
            entry["speaker_id"] = str(row[speaker_col])
        entries.append(entry)

    print(f"  Skipped: {missing_audio} missing audio, {bad_text} bad text, "
          f"{bad_audio} unreadable audio")
    print(f"  Loaded {len(entries)} entries "
          f"({sum(e['duration'] for e in entries) / 3600:.2f}h)")
    return entries


# ---------------------------------------------------------------------------
# Dataset & Collator
# ---------------------------------------------------------------------------

class GenericAudioDataset(Dataset):
    def __init__(self, entries: List[Dict], processor: Wav2Vec2Processor):
        self.entries = entries
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        start = item.get("start_time", 0.0)
        end = item.get("end_time")
        dur = (end - start) if end else None
        try:
            audio, _ = librosa.load(item["audio_path"], sr=16000,
                                    offset=start, duration=dur)
        except Exception as e:
            print(f"Error loading {item['audio_path']}: {e}")
            audio = np.zeros(16000, dtype=np.float32)
        iv = self.processor(audio, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(clean_text(item["text"])).input_ids
        return {"input_values": iv, "labels": labels}

class CTC_Trainer(Trainer):
    """Strip unsupported keys and force PEFT saves without embedding-layer export."""

    def _strip_inputs(self, inputs: dict) -> dict:
        allowed = {
            "input_values",
            "attention_mask",
            "labels",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        }
        return {k: v for k, v in inputs.items() if k in allowed}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return super().compute_loss(
            model,
            self._strip_inputs(inputs),
            return_outputs=return_outputs,
            **kwargs,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(
            model,
            self._strip_inputs(inputs),
            prediction_loss_only,
            ignore_keys,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save PEFT adapter safely for Wav2Vec2/CTC.
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                save_embedding_layers=False,
            )
        else:
            super()._save(output_dir, state_dict)

        # Keep tokenizer / processor with the checkpoint.
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Preserve TrainingArguments like the stock Trainer does.
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[str, bool] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad audio inputs explicitly via the feature extractor.
        # Calling self.processor.pad(...) is ambiguous in newer transformers
        # versions and can route through the tokenizer path.
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        # Pad labels via the tokenizer.
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on Nh of code-switching data. "
                    "Reads transcripts from the SwitchLingua_audio CSV directly."
    )
    parser.add_argument("--lang_pair", type=str, required=True, choices=ALL_LANG_PAIRS)
    parser.add_argument("--model_path", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to finetuned_lora_<lang_pair>.")
    parser.add_argument("--train_hours", type=float, default=1.0)
    parser.add_argument("--mucs_data_dir", type=str, default="mucs_data")
    parser.add_argument("--switchlingua_data_dir", type=str, default="switchlingua_data")
    parser.add_argument("--vocab_json", type=str, default=None)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip Hub round-trip; use cached local files only.")
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "out_proj"])
    # Training
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    if torch.cuda.is_available():
        _dev = "cuda"
    elif torch.backends.mps.is_available():
        _dev = "mps"
    else:
        _dev = "cpu"
    parser.add_argument("--device", type=str, default=_dev)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"finetuned_lora_{args.lang_pair}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading data: {args.lang_pair} (target train: {args.train_hours}h)")

    if args.lang_pair == "mucs_hin_eng":
        mucs_dir = Path(args.mucs_data_dir)
        extract_dir = download_mucs(
            mucs_dir, MUCS_HIN_ENG_URL, MUCS_HIN_ENG_FILENAME, "Hindi-English",
            skip_download=args.skip_download
        )
        all_entries = load_mucs_entries(extract_dir)
    else:
        lang_cfg = SWITCHLINGUA_LANGUAGES[args.lang_pair]
        sl_dir = Path(args.switchlingua_data_dir)
        sl_dir.mkdir(parents=True, exist_ok=True)
        sl_audio = download_switchlingua_audio(
            sl_dir, lang_cfg, skip_download=args.skip_download
        )
        all_entries = load_switchlingua_entries(sl_audio, lang_cfg)

    # CTC sanity: encoder produces ~1 frame per 320 samples at 16kHz.
    # Drop utterances where output frames < label length.
    valid = [
        e for e in all_entries
        if (e["duration"] * 16000 // 320) >= len(clean_text(e["text"]))
    ]
    n_dropped = len(all_entries) - len(valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} entries failing CTC length sanity check.")
    if not valid:
        raise RuntimeError("No valid entries after filtering.")

    total_hours = sum(e["duration"] for e in valid) / 3600.0
    print(f"  Total available: {len(valid)} entries, {total_hours:.2f}h")

    # ------------------------------------------------------------------
    # 2. Split into train / dev / test_pool
    # ------------------------------------------------------------------
    print(f"\n[2] Splitting train ({args.train_hours}h) + dev (~5min) + "
          f"test_pool (rest)...")

    rng = random.Random(args.seed)
    shuffled = valid[:]
    rng.shuffle(shuffled)

    train_entries, train_total = [], 0.0
    target_train_seconds = args.train_hours * 3600.0
    cursor = len(shuffled)
    for i, e in enumerate(shuffled):
        if train_total >= target_train_seconds:
            cursor = i
            break
        train_entries.append(e)
        train_total += e["duration"]

    leftover = shuffled[cursor:]
    if not leftover:
        raise RuntimeError(
            f"Not enough data: {total_hours:.2f}h total, "
            f"{args.train_hours}h requested. No data left for test_pool."
        )

    dev_entries, dev_total, test_entries = [], 0.0, []
    for e in leftover:
        if dev_total < 300.0:
            dev_entries.append(e)
            dev_total += e["duration"]
        else:
            test_entries.append(e)

    print(f"  Train:     {len(train_entries):>5} entries, {train_total / 3600:.2f}h")
    print(f"  Dev:       {len(dev_entries):>5} entries, {dev_total / 60:.1f} min  "
          f"(checkpointing only, NOT for reporting)")
    print(f"  Test pool: {len(test_entries):>5} entries, "
          f"{sum(e['duration'] for e in test_entries) / 3600:.2f}h "
          f"(saved for eval script)")

    # Disjointness check
    train_paths = {e["audio_path"] for e in train_entries}
    dev_paths = {e["audio_path"] for e in dev_entries}
    test_paths = {e["audio_path"] for e in test_entries}
    assert not (train_paths & dev_paths), "train/dev overlap"
    assert not (train_paths & test_paths), "train/test overlap"
    assert not (dev_paths & test_paths), "dev/test overlap"

    # ------------------------------------------------------------------
    # 3. Write manifests
    # ------------------------------------------------------------------
    for fname, items in [
        ("train_manifest.json", train_entries),
        ("dev_manifest.json", dev_entries),
        ("test_pool.json", test_entries),
    ]:
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {out_dir / fname} ({len(items)} entries)")

    # ------------------------------------------------------------------
    # 4. Processor / Vocab
    # ------------------------------------------------------------------
    print("\n[3] Setting up processor...")

    is_meta_checkpoint = (
        not args.model_path.startswith("facebook/")
        and not args.model_path.startswith("openai/")
        and os.path.isdir(args.model_path)
    )

    if is_meta_checkpoint:
        proc_path = (os.path.join(args.model_path, "processor")
                     if os.path.isdir(os.path.join(args.model_path, "processor"))
                     else args.model_path)
        model_load_path = (os.path.join(args.model_path, "meta_model")
                           if os.path.isdir(os.path.join(args.model_path, "meta_model"))
                           else args.model_path)
        print(f"  Loading meta processor from {proc_path}")
        processor = Wav2Vec2Processor.from_pretrained(proc_path)
        print(f"  Loading meta model from {model_load_path}")
        model = Wav2Vec2ForCTC.from_pretrained(
            model_load_path,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            ignore_mismatched_sizes=True,
        )
    else:
        proc_dir = out_dir / "processor"
        proc_dir.mkdir(parents=True, exist_ok=True)
        if args.vocab_json and os.path.exists(args.vocab_json):
            print(f"  Using provided vocab: {args.vocab_json}")
            import shutil
            shutil.copy(args.vocab_json, proc_dir / "vocab.json")
        else:
            print("  Building vocab from train + dev data only...")
            vocab = build_joint_vocab(
                [clean_text(e["text"]) for e in train_entries + dev_entries]
            )
            print(f"  Vocab size: {len(vocab)}")
            with open(proc_dir / "vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)

        tokenizer = Wav2Vec2CTCTokenizer(
            str(proc_dir / "vocab.json"),
            unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )
        processor.save_pretrained(proc_dir)

        print(f"  Loading base model {args.model_path} (new CTC head)...")
        model = Wav2Vec2ForCTC.from_pretrained(
            args.model_path,
            vocab_size=len(processor.tokenizer),
            pad_token_id=processor.tokenizer.pad_token_id,
            ctc_loss_reduction="mean",
            ignore_mismatched_sizes=True,
        )

    model.freeze_feature_encoder()

    # ------------------------------------------------------------------
    # 5. LoRA
    # ------------------------------------------------------------------
    print("\n[4] Applying LoRA...")
    print(f"  r={args.lora_r}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}, targets={args.lora_target_modules}")
    # lora_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     target_modules=args.lora_target_modules,
    #     bias="none",
    #     task_type=TaskType.FEATURE_EXTRACTION,
    # )
    lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_target_modules,
    bias="none",
    modules_to_save=["lm_head"],  # important for CTC
)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(args.device)

    # ------------------------------------------------------------------
    # 6. Datasets + Trainer
    # ------------------------------------------------------------------
    train_dataset = GenericAudioDataset(train_entries, processor)
    dev_dataset = GenericAudioDataset(dev_entries, processor)
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Critical for Wav2Vec2: don't let the Trainer drop columns based on
        # the model's forward() signature. peft wraps the model and the column
        # introspection can drop input_values or rename keys.
        remove_unused_columns=False,
        dataloader_num_workers=0,
        push_to_hub=False,
        report_to="none",
        seed=args.seed,
        fp16=(args.device == "cuda"),
    )

    trainer = CTC_Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # processing_class=processor.feature_extractor,
    )

    print(f"\n[5] Training on {args.lang_pair} ({train_total / 3600:.2f}h)...")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    # print(f"\n[6] Saving adapter to {args.output_dir}...")
    # trainer.save_model(args.output_dir)
    # processor.save_pretrained(args.output_dir)

    # merged_dir = os.path.join(args.output_dir, "merged")
    # print(f"  Merging LoRA -> {merged_dir}")
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(merged_dir)
    # processor.save_pretrained(merged_dir)
    print(f"\n[6] Saving adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir, save_embedding_layers=False)
    processor.save_pretrained(args.output_dir)

    merged_dir = os.path.join(args.output_dir, "merged")
    print(f"  Merging LoRA -> {merged_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    meta = {
        "lang_pair": args.lang_pair,
        "train_hours": train_total / 3600.0,
        "n_train": len(train_entries),
        "n_dev": len(dev_entries),
        "n_test_pool": len(test_entries),
        "test_pool_hours": sum(e["duration"] for e in test_entries) / 3600.0,
        "model_path": args.model_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "seed": args.seed,
        "manifests": {
            "train": str(out_dir / "train_manifest.json"),
            "dev": str(out_dir / "dev_manifest.json"),
            "test_pool": str(out_dir / "test_pool.json"),
        },
    }
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Run metadata: {out_dir / 'train_meta.json'}")


if __name__ == "__main__":
    main()