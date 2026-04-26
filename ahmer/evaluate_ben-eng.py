"""
Evaluation script for a fine-tuned Wav2Vec2ForCTC model on:
  1. Local CS-FLEURS Bengali-English test set (manifest.csv format)
  2. OpenSLR104 real Bengali-English test set (Kaldi format)

Either evaluation can be run independently: if you only pass the
local args, only local eval runs; if you only pass the OpenSLR args,
only that one runs. Makes debugging easier.

Usage — both datasets:
  python ahmer/evaluate_ben-eng.py \
    --model_path runs/w2v2_csfleurs_ben_eng_fix3 \
    --device cuda \
    --local_test_dir data/cs_fleurs_ben_eng_300_49/test \
    --real_wav_root test-SLR-ben \
    --real_wav_scp test-SLR-ben/transcripts/wav.scp \
    --real_segments test-SLR-ben/transcripts/segments \
    --real_text test-SLR-ben/transcripts/text \
    --save_json runs/w2v2_csfleurs_ben_eng_fix3/eval_results.json

Usage — OpenSLR only (for debugging):
  python ahmer/evaluate_ben-eng.py \
    --model_path runs/w2v2_csfleurs_ben_eng_fix3 \
    --real_wav_root test-SLR-ben \
    --real_wav_scp test-SLR-ben/transcripts/wav.scp \
    --real_segments test-SLR-ben/transcripts/segments \
    --real_text test-SLR-ben/transcripts/text
"""

import os
import re
import json
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from evaluate import load as load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text cleaning — MUST match training script exactly
# ---------------------------------------------------------------------------
_KEEP_PATTERN = re.compile(
    r"[^"
    r"\u0980-\u09FF"   # Bengali block
    r"A-Za-z"          # Latin letters
    r"0-9"             # digits
    r" "               # space
    r"]"
)


def clean_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("**", "").replace("__", "")
    text = text.lower()
    text = _KEEP_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Model / inference helpers
# ---------------------------------------------------------------------------
def load_model_and_processor(model_path: str, device: str):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Align config with tokenizer (no bos/eos)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = None
    model.config.eos_token_id = None

    model.to(device)
    model.eval()
    return model, processor


def transcribe_audio(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    audio: np.ndarray,
    device: str,
    target_sr: int = 16000,
) -> str:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    inputs = processor(
        audio, sampling_rate=target_sr, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(device)
    attention_mask = (
        inputs.attention_mask.to(device)
        if "attention_mask" in inputs else None
    )

    with torch.no_grad():
        outputs = model(
            input_values=input_values, attention_mask=attention_mask
        )
        pred_ids = torch.argmax(outputs.logits, dim=-1)

    pred = processor.batch_decode(pred_ids)[0]
    return clean_text(pred)


def compute_wer_cer(
    predictions: List[str], references: List[str]
) -> Tuple[float, float]:
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    # Guard against empty strings which crash the metric
    predictions = [p if p.strip() else " " for p in predictions]
    references = [r if r.strip() else " " for r in references]
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    return wer, cer


# ---------------------------------------------------------------------------
# Local CS-FLEURS evaluation
# ---------------------------------------------------------------------------
def evaluate_local_csfleurs_test(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: str,
    test_dir: str,
    max_samples: Optional[int] = None,
) -> Tuple[Dict, pd.DataFrame]:
    manifest_path = os.path.join(test_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    rows = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CS-FLEURS test"):
        rel_audio = str(row["audio"]).replace("\\", "/")
        audio_path = os.path.join(test_dir, rel_audio)
        ref = clean_text(row["text"])

        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            pred = transcribe_audio(model, processor, audio, device)
            rows.append({
                "audio_path": rel_audio, "reference": ref, "prediction": pred
            })
            if len(rows) <= 5:
                print(f"  REF : {ref}")
                print(f"  PRED: {pred}")
                print("  " + "-" * 78)
        except Exception as e:
            skipped += 1
            print(f"[CS-FLEURS] Skipping {audio_path}: {e}")

    if not rows:
        raise RuntimeError("No CS-FLEURS samples were successfully evaluated.")

    result_df = pd.DataFrame(rows)
    wer, cer = compute_wer_cer(
        result_df["prediction"].tolist(), result_df["reference"].tolist()
    )
    return {
        "dataset": "local_csfleurs_ben_eng_test",
        "num_scored": len(rows),
        "num_skipped": skipped,
        "wer": float(wer),
        "cer": float(cer),
    }, result_df


# ---------------------------------------------------------------------------
# Kaldi file readers — with diagnostics
# ---------------------------------------------------------------------------
def read_kaldi_wav_scp(path: str) -> Dict[str, str]:
    """Format: <recording_id> <wav_path_or_pipe>"""
    mapping = {}
    skipped_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                skipped_lines += 1
                if skipped_lines <= 3:
                    print(f"  [wav.scp] bad line {line_no}: {line!r}")
                continue
            rec_id, wav_path = parts
            # Handle piped commands (Kaldi "command |" format) — warn but skip
            if wav_path.rstrip().endswith("|"):
                skipped_lines += 1
                if skipped_lines <= 3:
                    print(
                        f"  [wav.scp] line {line_no} uses pipe command "
                        f"(not supported by this script): {wav_path!r}"
                    )
                continue
            mapping[rec_id] = wav_path
    print(f"  Loaded {len(mapping)} wav.scp entries "
          f"(skipped {skipped_lines} bad lines)")
    return mapping


def read_kaldi_segments(path: str) -> List[Tuple[str, str, float, float]]:
    """Format: <utt_id> <rec_id> <start_sec> <end_sec>"""
    segments = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                skipped += 1
                if skipped <= 3:
                    print(f"  [segments] bad line {line_no} "
                          f"(expected 4 fields, got {len(parts)}): {line!r}")
                continue
            utt_id, rec_id, start, end = parts
            try:
                segments.append((utt_id, rec_id, float(start), float(end)))
            except ValueError:
                skipped += 1
                if skipped <= 3:
                    print(f"  [segments] non-numeric times at line {line_no}: "
                          f"{line!r}")
    print(f"  Loaded {len(segments)} segments "
          f"(skipped {skipped} bad lines)")
    return segments


def read_kaldi_text(path: str) -> Dict[str, str]:
    """Format: <utt_id> <transcript>"""
    texts = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                skipped += 1
                if skipped <= 3:
                    print(f"  [text] bad line {line_no}: {line!r}")
                continue
            utt_id, text = parts
            texts[utt_id] = text
    print(f"  Loaded {len(texts)} transcripts "
          f"(skipped {skipped} bad lines)")
    return texts


def resolve_audio_path(wav_root: str, wav_rel_or_name: str) -> str:
    """Resolve a wav.scp entry to an absolute path."""
    if os.path.isabs(wav_rel_or_name):
        return wav_rel_or_name
    return os.path.join(wav_root, wav_rel_or_name)


# ---------------------------------------------------------------------------
# OpenSLR104 evaluation
# ---------------------------------------------------------------------------
def evaluate_openslr104_kaldi(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: str,
    wav_root: str,
    wav_scp_path: str,
    segments_path: str,
    text_path: str,
    max_samples: Optional[int] = None,
    dataset_name: str = "openslr104_ben_eng_real_test",
) -> Tuple[Dict, pd.DataFrame]:
    print("\n[Reading Kaldi files]")
    for label, p in [("wav.scp", wav_scp_path),
                     ("segments", segments_path),
                     ("text", text_path)]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {label} file: {p}")
        print(f"  {label}: {p} (size={os.path.getsize(p)} bytes)")

    wav_map = read_kaldi_wav_scp(wav_scp_path)
    segments = read_kaldi_segments(segments_path)
    text_map = read_kaldi_text(text_path)

    # Sanity-check the wav_root: try to find the first wav file
    if wav_map:
        first_rec_id = next(iter(wav_map))
        first_wav_rel = wav_map[first_rec_id]
        first_wav_path = resolve_audio_path(wav_root, first_wav_rel)
        print(f"\n[wav_root sanity check]")
        print(f"  First wav.scp entry : {first_rec_id} -> {first_wav_rel!r}")
        print(f"  Resolved to         : {first_wav_path}")
        print(f"  Exists              : {os.path.exists(first_wav_path)}")
        if not os.path.exists(first_wav_path):
            # Try with just the basename in wav_root
            alt = os.path.join(wav_root, os.path.basename(first_wav_rel))
            print(f"  Alt (basename only) : {alt}")
            print(f"  Alt exists          : {os.path.exists(alt)}")

    if max_samples is not None:
        segments = segments[:max_samples]

    rows = []
    skipped_missing_text = 0
    skipped_missing_wav = 0
    skipped_load_error = 0

    for utt_id, rec_id, start, end in tqdm(
        segments, desc="OpenSLR104 ben-eng test"
    ):
        if utt_id not in text_map:
            skipped_missing_text += 1
            continue
        if rec_id not in wav_map:
            skipped_missing_wav += 1
            continue

        audio_path = resolve_audio_path(wav_root, wav_map[rec_id])
        # If not found directly, try looking in wav_root by basename
        if not os.path.exists(audio_path):
            alt = os.path.join(wav_root, os.path.basename(wav_map[rec_id]))
            if os.path.exists(alt):
                audio_path = alt

        try:
            duration = max(0.0, end - start)
            audio, _ = librosa.load(
                audio_path, sr=16000, offset=start, duration=duration
            )
            if audio is None or len(audio) == 0:
                raise RuntimeError("empty segment")
            pred = transcribe_audio(model, processor, audio, device)
            ref = clean_text(text_map[utt_id])
            rows.append({
                "utt_id": utt_id, "rec_id": rec_id,
                "audio_path": audio_path,
                "start": start, "end": end,
                "reference": ref, "prediction": pred,
            })
            if len(rows) <= 5:
                print(f"  REF : {ref}")
                print(f"  PRED: {pred}")
                print("  " + "-" * 78)
        except Exception as e:
            skipped_load_error += 1
            if skipped_load_error <= 5:
                print(f"[OpenSLR104] Skipping {utt_id} "
                      f"from {audio_path}: {e}")

    total_skipped = (skipped_missing_text + skipped_missing_wav
                     + skipped_load_error)
    print(f"\n[OpenSLR104 summary]")
    print(f"  Scored      : {len(rows)}")
    print(f"  Skipped     : {total_skipped}")
    print(f"    - no text : {skipped_missing_text}")
    print(f"    - no wav  : {skipped_missing_wav}")
    print(f"    - load err: {skipped_load_error}")

    if not rows:
        raise RuntimeError(
            "No OpenSLR104 samples were successfully evaluated. "
            "Check the diagnostic output above for root cause: "
            "bad file formats, missing wavs, or wav_root mismatch."
        )

    result_df = pd.DataFrame(rows)
    wer, cer = compute_wer_cer(
        result_df["prediction"].tolist(), result_df["reference"].tolist()
    )
    return {
        "dataset": dataset_name,
        "num_scored": len(rows),
        "num_skipped": total_skipped,
        "skipped_missing_text": skipped_missing_text,
        "skipped_missing_wav": skipped_missing_wav,
        "skipped_load_error": skipped_load_error,
        "wer": float(wer),
        "cer": float(cer),
    }, result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"]
    )

    # Local CS-FLEURS args (optional)
    parser.add_argument("--local_test_dir", type=str, default=None,
                        help="Path to CS-FLEURS test dir with manifest.csv")
    parser.add_argument("--max_samples_local", type=int, default=None)

    # OpenSLR104 args (optional — pass together)
    parser.add_argument("--real_wav_root", type=str, default=None)
    parser.add_argument("--real_wav_scp", type=str, default=None)
    parser.add_argument("--real_segments", type=str, default=None)
    parser.add_argument("--real_text", type=str, default=None)
    parser.add_argument("--max_samples_real", type=int, default=None)

    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="Save per-sample predictions as CSV")
    args = parser.parse_args()

    # Validation: OpenSLR args must be all-or-nothing
    openslr_args = [
        args.real_wav_root, args.real_wav_scp,
        args.real_segments, args.real_text
    ]
    have_some = any(a is not None for a in openslr_args)
    have_all = all(a is not None for a in openslr_args)
    if have_some and not have_all:
        parser.error(
            "OpenSLR evaluation requires ALL of: --real_wav_root, "
            "--real_wav_scp, --real_segments, --real_text"
        )

    if not args.local_test_dir and not have_all:
        parser.error(
            "Provide --local_test_dir and/or the four --real_* arguments "
            "(nothing to evaluate otherwise)."
        )

    print(f"Loading model from: {args.model_path}")
    model, processor = load_model_and_processor(args.model_path, args.device)
    print(f"Vocab size: {len(processor.tokenizer)}")
    print(f"Pad token id: {processor.tokenizer.pad_token_id}")

    results = {"model_path": args.model_path}

    # -----------------------------------------------------------------
    # Local CS-FLEURS
    # -----------------------------------------------------------------
    if args.local_test_dir:
        print("\n" + "=" * 70)
        print("Evaluating local CS-FLEURS ben-eng test")
        print("=" * 70)
        local_result, local_df = evaluate_local_csfleurs_test(
            model=model, processor=processor, device=args.device,
            test_dir=args.local_test_dir,
            max_samples=args.max_samples_local,
        )
        print("\n" + json.dumps(local_result, ensure_ascii=False, indent=2))
        results["local_csfleurs_test"] = local_result

        if args.save_predictions:
            out_csv = os.path.join(
                args.model_path, "eval_csfleurs_predictions.csv"
            )
            local_df.to_csv(out_csv, index=False)
            print(f"  Per-sample predictions saved to: {out_csv}")

    # -----------------------------------------------------------------
    # OpenSLR104
    # -----------------------------------------------------------------
    if have_all:
        print("\n" + "=" * 70)
        print("Evaluating OpenSLR104 Bengali-English real test")
        print("=" * 70)
        real_result, real_df = evaluate_openslr104_kaldi(
            model=model, processor=processor, device=args.device,
            wav_root=args.real_wav_root,
            wav_scp_path=args.real_wav_scp,
            segments_path=args.real_segments,
            text_path=args.real_text,
            max_samples=args.max_samples_real,
        )
        print("\n" + json.dumps(real_result, ensure_ascii=False, indent=2))
        results["openslr104_real"] = real_result

        if args.save_predictions:
            out_csv = os.path.join(
                args.model_path, "eval_openslr104_predictions.csv"
            )
            real_df.to_csv(out_csv, index=False)
            print(f"  Per-sample predictions saved to: {out_csv}")

    # -----------------------------------------------------------------
    # Final summary + save
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for key, val in results.items():
        if key == "model_path":
            continue
        print(f"\n{key}:")
        print(f"  WER     : {val['wer']:.4f}")
        print(f"  CER     : {val['cer']:.4f}")
        print(f"  Scored  : {val['num_scored']}")
        print(f"  Skipped : {val['num_skipped']}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()


# import os
# import json
# import argparse
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional

# import librosa
# import numpy as np
# import pandas as pd
# import torch
# from evaluate import load
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# from tqdm import tqdm


# def clean_text(text: str) -> str:
#     return str(text).replace("**", "").replace("__", "").strip()


# def load_model_and_processor(model_path: str, device: str):
#     processor = Wav2Vec2Processor.from_pretrained(model_path)
#     model = Wav2Vec2ForCTC.from_pretrained(model_path)
#     model.to(device)
#     model.eval()
#     return model, processor


# def transcribe_audio(
#     model: Wav2Vec2ForCTC,
#     processor: Wav2Vec2Processor,
#     audio: np.ndarray,
#     device: str,
#     target_sr: int = 16000,
# ) -> str:
#     if audio.ndim > 1:
#         audio = np.mean(audio, axis=1)
#     audio = audio.astype(np.float32)

#     inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
#     input_values = inputs.input_values.to(device)
#     attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

#     with torch.no_grad():
#         outputs = model(input_values=input_values, attention_mask=attention_mask)
#         pred_ids = torch.argmax(outputs.logits, dim=-1)

#     pred = processor.batch_decode(pred_ids)[0]
#     return clean_text(pred)


# def compute_wer_cer(predictions: List[str], references: List[str]) -> Tuple[float, float]:
#     wer_metric = load("wer")
#     cer_metric = load("cer")
#     wer = wer_metric.compute(predictions=predictions, references=references)
#     cer = cer_metric.compute(predictions=predictions, references=references)
#     return wer, cer


# def evaluate_local_csfleurs_test(
#     model: Wav2Vec2ForCTC,
#     processor: Wav2Vec2Processor,
#     device: str,
#     test_dir: str,
#     max_samples: Optional[int] = None,
# ) -> Dict:
#     manifest_path = os.path.join(test_dir, "manifest.csv")
#     if not os.path.exists(manifest_path):
#         raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

#     df = pd.read_csv(manifest_path)

#     if max_samples is not None:
#         df = df.iloc[:max_samples].reset_index(drop=True)

#     predictions = []
#     references = []
#     skipped = 0

#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Local CS-FLEURS ben-eng test"):
#         rel_audio = str(row["audio"]).replace("\\", "/")
#         audio_path = os.path.join(test_dir, rel_audio)
#         ref = clean_text(row["text"])

#         try:
#             audio, _ = librosa.load(audio_path, sr=16000)
#             pred = transcribe_audio(model, processor, audio, device, target_sr=16000)
#             predictions.append(pred)
#             references.append(ref)
#             if len(predictions) < 10:
#                 print(f"REF : {ref}")
#                 print(f"PRED: {pred}")
#                 print("-" * 80)
#         except Exception as e:
#             skipped += 1
#             print(f"[Local CS-FLEURS] Skipping {audio_path}: {e}")

#     if not predictions:
#         raise RuntimeError("No local CS-FLEURS test samples were successfully evaluated.")

#     wer, cer = compute_wer_cer(predictions, references)
#     return {
#         "dataset": "local_csfleurs_ben_eng_test",
#         "num_scored": len(predictions),
#         "num_skipped": skipped,
#         "wer": wer,
#         "cer": cer,
#     }


# def read_kaldi_wav_scp(path: str) -> Dict[str, str]:
#     mapping = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.rstrip("\n").split(maxsplit=1)
#             if len(parts) != 2:
#                 continue
#             rec_id, wav_path = parts
#             mapping[rec_id] = wav_path
#     return mapping


# def read_kaldi_segments(path: str) -> List[Tuple[str, str, float, float]]:
#     segments = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 4:
#                 continue
#             utt_id, rec_id, start, end = parts
#             segments.append((utt_id, rec_id, float(start), float(end)))
#     return segments


# def read_kaldi_text(path: str) -> Dict[str, str]:
#     texts = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.rstrip("\n").split(maxsplit=1)
#             if len(parts) != 2:
#                 continue
#             utt_id, text = parts
#             texts[utt_id] = text
#     return texts


# def resolve_audio_path(wav_root: str, wav_rel_or_name: str) -> str:
#     if os.path.isabs(wav_rel_or_name):
#         return wav_rel_or_name
#     return os.path.join(wav_root, wav_rel_or_name)


# def evaluate_openslr104_kaldi(
#     model: Wav2Vec2ForCTC,
#     processor: Wav2Vec2Processor,
#     device: str,
#     wav_root: str,
#     wav_scp_path: str,
#     segments_path: str,
#     text_path: str,
#     max_samples: Optional[int] = None,
#     dataset_name: str = "openslr104_ben_eng_real_test",
# ) -> Dict:
#     wav_map = read_kaldi_wav_scp(wav_scp_path)
#     segments = read_kaldi_segments(segments_path)
#     text_map = read_kaldi_text(text_path)

#     if max_samples is not None:
#         segments = segments[:max_samples]

#     predictions = []
#     references = []
#     skipped = 0

#     for utt_id, rec_id, start, end in tqdm(segments, desc="OpenSLR104 ben-eng test"):
#         if utt_id not in text_map:
#             skipped += 1
#             print(f"[OpenSLR104] Missing transcript for utterance: {utt_id}")
#             continue
#         if rec_id not in wav_map:
#             skipped += 1
#             print(f"[OpenSLR104] Missing wav.scp entry for recording: {rec_id}")
#             continue

#         audio_path = resolve_audio_path(wav_root, wav_map[rec_id])

#         try:
#             duration = max(0.0, end - start)
#             audio, _ = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
#             if audio is None or len(audio) == 0:
#                 raise RuntimeError("empty segment")
#             pred = transcribe_audio(model, processor, audio, device, target_sr=16000)
#             ref = clean_text(text_map[utt_id])
#             predictions.append(pred)
#             references.append(ref)
#             if len(predictions) < 10:
#                 print(f"REF : {ref}")
#                 print(f"PRED: {pred}")
#                 print("-" * 80)
#         except Exception as e:
#             skipped += 1
#             print(f"[OpenSLR104] Skipping {utt_id} from {audio_path}: {e}")

#     if not predictions:
#         raise RuntimeError("No OpenSLR104 samples were successfully evaluated.")

#     wer, cer = compute_wer_cer(predictions, references)
#     return {
#         "dataset": dataset_name,
#         "num_scored": len(predictions),
#         "num_skipped": skipped,
#         "wer": wer,
#         "cer": cer,
#     }


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model directory")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])

#     parser.add_argument(
#         "--local_test_dir",
#         type=str,
#         required=True,
#         help="Path to local CS-FLEURS ben-eng test directory containing manifest.csv and audio/",
#     )
#     parser.add_argument("--max_samples_local", type=int, default=None)

#     parser.add_argument("--real_wav_root", type=str, required=True, help="Directory containing extracted OpenSLR104 WAV files")
#     parser.add_argument("--real_wav_scp", type=str, required=True, help="Path to wav.scp")
#     parser.add_argument("--real_segments", type=str, required=True, help="Path to segments")
#     parser.add_argument("--real_text", type=str, required=True, help="Path to text")
#     parser.add_argument("--max_samples_real", type=int, default=None)

#     parser.add_argument("--save_json", type=str, default=None, help="Optional path to save results JSON")
#     args = parser.parse_args()

#     print(f"Loading model from: {args.model_path}")
#     model, processor = load_model_and_processor(args.model_path, args.device)

#     print("\n--- Evaluating local CS-FLEURS ben-eng test ---")
#     local_result = evaluate_local_csfleurs_test(
#         model=model,
#         processor=processor,
#         device=args.device,
#         test_dir=args.local_test_dir,
#         max_samples=args.max_samples_local,
#     )
#     print(json.dumps(local_result, ensure_ascii=False, indent=2))

#     print("\n--- Evaluating OpenSLR104 Bengali-English test ---")
#     real_result = evaluate_openslr104_kaldi(
#         model=model,
#         processor=processor,
#         device=args.device,
#         wav_root=args.real_wav_root,
#         wav_scp_path=args.real_wav_scp,
#         segments_path=args.real_segments,
#         text_path=args.real_text,
#         max_samples=args.max_samples_real,
#         dataset_name="openslr104_ben_eng_real_test",
#     )
#     print(json.dumps(real_result, ensure_ascii=False, indent=2))

#     results = {
#         "model_path": args.model_path,
#         "local_csfleurs_test": local_result,
#         "openslr104_real": real_result,
#     }

#     if args.save_json:
#         out_path = Path(args.save_json)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
#         print(f"\nSaved results to: {out_path}")


# if __name__ == "__main__":
#     main()


# """
# Example:
# python ahmer/evaluate_ben-eng.py \
#   --model_path /workspace/projects/asr_project_src/runs/w2v2_csfleurs_ben_eng \
#   --device cuda \
#   --local_test_dir /workspace/projects/asr_project_src/data/cs_fleurs_ben_eng_300_49/test \
#   --real_wav_root /workspace/projects/asr_project_src/test-SLR-ben \
#   --real_wav_scp /workspace/projects/asr_project_src/test-SLR-ben/transcripts/wav.scp \
#   --real_segments /workspace/projects/asr_project_src/test-SLR-ben/transcripts/segments \
#   --real_text /workspace/projects/asr_project_src/test-SLR-ben/transcripts/text \
#   --save_json /workspace/projects/asr_project_src/runs/w2v2_csfleurs_ben_eng/eval_results.json


# python ahmer/evaluate_ben-eng.py \
#   --model_path /workspace/projects/asr_project_src/runs/w2v2_csfleurs_ben_eng_using_test_as_validation \
#   --device cuda \
#   --local_test_dir /workspace/projects/asr_project_src/data/cs_fleurs_ben_eng_300_49/test \
#   --real_wav_root /workspace/projects/asr_project_src/test-SLR-ben \
#   --real_wav_scp /workspace/projects/asr_project_src/test-SLR-ben/transcripts/wav.scp \
#   --real_segments /workspace/projects/asr_project_src/test-SLR-ben/transcripts/segments \
#   --real_text /workspace/projects/asr_project_src/test-SLR-ben/transcripts/text \
#   --save_json /workspace/projects/asr_project_src/runs/w2v2_csfleurs_ben_eng_using_test_as_validation/eval_results.json
# """