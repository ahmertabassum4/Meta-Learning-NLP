"""
Finetune facebook/wav2vec2-xls-r-300m on Hindi-English code-switched audio using LoRA.

All known issues fixed:
  - Blank-collapse prevention (ctc_zero_infinity, lm_head re-init, FFN LoRA targets)
  - Correct compute_metrics (skip_special_tokens, tokenizer.batch_decode)
  - Space → | normalisation in both vocab build and __getitem__
  - processing_class=processor (full processor, not just feature_extractor)
  - Extended LoRA targets to include FFN layers
  - Data cap raised to 3–4 hours (pass --max_hours 3.0 or 4.0)
  - Lower LR (1e-4) + cosine scheduler + longer warmup
  - Gradient-flow verification after PEFT wrapping
  - eval_accumulation_steps to avoid OOM during eval on long runs

Expected data layout (Kaldi-style):

    <data_root>/
        _meta_full/train/transcripts/
            text         # <utt_id> <transcription>
            wav.scp      # <recording_id> <path/to/wav>
            segments     # <utt_id> <recording_id> <start_sec> <end_sec>
            utt2spk      # (unused)
        audio/
            *.wav

Example:
    python finetune_xlsr_lora_hineng.py \
        --data_root /workspace/projects/asr_project_src/hineng_cs_1hr \
        --output_dir runs/xlsr_lora_hineng_4hr \
        --max_hours 4.0 \
        --num_train_epochs 60 \
        --batch_size 8 \
        --grad_accum 4 \
        --learning_rate 1e-4
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import evaluate


# ============================================================================ #
#  Kaldi file parsing                                                           #
# ============================================================================ #

def read_kaldi_two_col(path: str) -> Dict[str, str]:
    """Read a Kaldi file where each line is `<key> <rest_of_line>`."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            out[parts[0]] = parts[1] if len(parts) == 2 else ""
    return out


def read_segments(path: str) -> Dict[str, Dict]:
    """segments: <utt_id> <rec_id> <start_sec> <end_sec>"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            utt_id, rec_id = parts[0], parts[1]
            start, end = float(parts[2]), float(parts[3])
            out[utt_id] = {"rec_id": rec_id, "start": start, "end": end}
    return out


def _index_audio_dir(audio_dir: str) -> Dict[str, str]:
    """Map every basename (with and without extension) → full path."""
    index = {}
    if not os.path.isdir(audio_dir):
        return index
    for root, _, files in os.walk(audio_dir):
        for fn in files:
            full = os.path.join(root, fn)
            index[fn] = full
            stem, _ = os.path.splitext(fn)
            index.setdefault(stem, full)
    return index


def resolve_wav_path(
    scp_value: str,
    data_root: str,
    audio_index: Dict[str, str],
) -> Union[str, None]:
    val = scp_value.strip()
    if not val:
        return None

    # Kaldi pipe — try to salvage a filename from the command string
    if val.endswith("|"):
        for tok in val.split():
            if tok.endswith(".wav") or tok.endswith(".flac"):
                hit = audio_index.get(os.path.basename(tok))
                if hit:
                    return hit
        return None

    if os.path.isabs(val) and os.path.exists(val):
        return val
    cand = os.path.join(data_root, val)
    if os.path.exists(cand):
        return cand

    base = os.path.basename(val)
    return audio_index.get(base) or audio_index.get(os.path.splitext(base)[0])


def build_entries(
    data_root: str,
    transcripts_dir: str,
    strict: bool = False,
) -> List[Dict]:
    tdir = os.path.join(data_root, transcripts_dir)
    text    = read_kaldi_two_col(os.path.join(tdir, "text"))
    wav_scp = read_kaldi_two_col(os.path.join(tdir, "wav.scp"))

    segments_path = os.path.join(tdir, "segments")
    has_segments  = os.path.exists(segments_path)
    segments      = read_segments(segments_path) if has_segments else {}

    audio_index = _index_audio_dir(os.path.join(data_root, "audio"))
    print(f"[data] Indexed {len(audio_index)} audio basenames under {data_root}/audio")

    entries             = []
    missing_seg         = 0
    missing_rec_in_scp  = 0
    unresolved_path     = 0
    bad_examples: List[str] = []

    for utt_id, transcription in text.items():
        if has_segments:
            seg = segments.get(utt_id)
            if seg is None:
                missing_seg += 1
                continue
            rec_id       = seg["rec_id"]
            start, end   = seg["start"], seg["end"]
        else:
            rec_id     = utt_id
            start, end = None, None

        wav_val = wav_scp.get(rec_id)
        if wav_val is None:
            missing_rec_in_scp += 1
            continue

        wav_path = resolve_wav_path(wav_val, data_root, audio_index)
        if wav_path is None:
            unresolved_path += 1
            if len(bad_examples) < 3:
                bad_examples.append(
                    f"  utt={utt_id}  rec={rec_id}  scp_value={wav_val!r}"
                )
            continue

        entries.append(
            dict(utt_id=utt_id, text=transcription,
                 wav_path=wav_path, start=start, end=end)
        )

    dropped = missing_seg + missing_rec_in_scp + unresolved_path
    print(
        f"[data] Resolved {len(entries)}/{len(text)} utterances "
        f"(dropped: missing_seg={missing_seg}, "
        f"rec_not_in_scp={missing_rec_in_scp}, "
        f"file_not_found={unresolved_path})"
    )
    if bad_examples:
        print("[data] Examples of unresolved paths:")
        for ex in bad_examples:
            print(ex)

    if strict and dropped > 0:
        raise RuntimeError(
            f"--strict set and {dropped} utterances could not be resolved."
        )
    if not entries:
        raise RuntimeError(
            "Zero usable utterances. Check wav.scp recording IDs vs audio/ filenames."
        )
    return entries


# ============================================================================ #
#  Text helpers                                                                 #
# ============================================================================ #

def clean_text(text: str) -> str:
    """Strip markdown artefacts and leading/trailing whitespace."""
    return text.replace("**", "").replace("__", "").strip()


def normalize_text(text: str) -> str:
    """clean + replace space with | (Wav2Vec2 word-delimiter convention)."""
    return clean_text(text).replace(" ", "|")


def build_vocab(texts: List[str]) -> Dict[str, int]:
    """
    Character-level vocab from training transcripts.
    Devanagari + Latin + digits + punctuation — no language-specific rules.
    Space is folded into '|' (word delimiter) before indexing.
    """
    all_chars: set = set()
    for t in texts:
        all_chars.update(normalize_text(t))
    all_chars.discard(" ")          # should already be gone, but be safe

    vocab = {c: i for i, c in enumerate(sorted(all_chars))}
    vocab.setdefault("|", len(vocab))   # word delimiter — must exist
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


# ============================================================================ #
#  Dataset                                                                      #
# ============================================================================ #

class HinEngKaldiDataset(Dataset):
    def __init__(
        self,
        entries: List[Dict],
        processor: Wav2Vec2Processor,
        sr: int = 16000,
    ):
        self.entries   = entries
        self.processor = processor
        self.sr        = sr

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        try:
            if e["start"] is not None:
                audio, _ = librosa.load(
                    e["wav_path"], sr=self.sr,
                    offset=e["start"], duration=e["end"] - e["start"],
                )
            else:
                audio, _ = librosa.load(e["wav_path"], sr=self.sr)
        except Exception as err:
            raise RuntimeError(
                f"Failed to load {e['wav_path']} (utt={e['utt_id']}): {err}"
            ) from err

        input_values = self.processor.feature_extractor(
            audio, sampling_rate=self.sr
        ).input_values[0]

        # FIX: normalize spaces → | before tokenising
        labels = self.processor.tokenizer(normalize_text(e["text"])).input_ids
        return {"input_values": input_values, "labels": labels}


# ============================================================================ #
#  Data collator                                                                #
# ============================================================================ #

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}          for f in features]

        # Audio padding via feature_extractor
        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        # Label padding via tokenizer
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        # Replace pad positions with -100 so CTC loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ============================================================================ #
#  Custom Trainer (PEFT-aware)                                                 #
# ============================================================================ #

class PeftAudioTrainer(Trainer):
    """
    Two responsibilities beyond stock Trainer:
      1. Save only LoRA adapter weights (no missing embedding-layer errors).
      2. Provide a correct prediction_step because PeftModel.forward(**kwargs)
         has a generic signature that confuses Trainer's argument-sniffing.
    """

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs     = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(
                    input_values   = inputs["input_values"],
                    attention_mask = inputs.get("attention_mask"),
                    labels         = inputs.get("labels"),
                )
            loss   = outputs.loss.detach() if has_labels else None
            logits = outputs.logits.detach()
            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.detach()

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


# ============================================================================ #
#  Metrics                                                                      #
# ============================================================================ #

def make_compute_metrics(processor: Wav2Vec2Processor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        # Cast to fp32 — bf16 numpy arrays can break argmax in older numpy
        pred_logits = np.asarray(pred.predictions, dtype=np.float32)
        pred_ids    = np.argmax(pred_logits, axis=-1)

        label_ids = np.array(pred.label_ids, copy=True)
        # Replace -100 fill with pad_token_id so the tokenizer can decode
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # FIX: use tokenizer.batch_decode with skip_special_tokens=True
        # so [PAD] / [UNK] never appear in the decoded strings
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        ref_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Debug: show a few REF / PRED pairs every eval round
        print("\n[eval] Sample predictions:")
        for p, r in zip(pred_str[:4], ref_str[:4]):
            print(f"  REF : {r!r}")
            print(f"  PRED: {p!r}")
        print()

        # Drop pairs where reference is empty (would crash WER)
        pairs = [(p, r) for p, r in zip(pred_str, ref_str) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        preds, refs = zip(*pairs)

        wer = wer_metric.compute(predictions=list(preds), references=list(refs))
        cer = cer_metric.compute(predictions=list(preds), references=list(refs))
        return {"wer": round(wer, 4), "cer": round(cer, 4)}

    return compute_metrics


# ============================================================================ #
#  Helpers                                                                      #
# ============================================================================ #

def print_lora_target_candidates(model: torch.nn.Module) -> None:
    """Print all Linear layer name-suffixes so the user can pick LoRA targets."""
    print("\n[lora] Linear modules available as LoRA targets (use the suffix after last '.'):")
    seen_suffixes = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.rsplit(".", 1)[-1]
            if suffix not in seen_suffixes:
                print(f"  {name}  →  suffix='{suffix}'")
                seen_suffixes.add(suffix)
    print()


def verify_trainable_params(model: torch.nn.Module) -> None:
    """Print every trainable parameter so we can confirm lm_head is included."""
    print("[lora] Trainable parameters after PEFT wrapping:")
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:80s}  {list(param.shape)}")
            total += param.numel()
    print(f"  → Total trainable: {total:,}\n")


# ============================================================================ #
#  Main                                                                         #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune XLS-R 300M on Hindi-English CS audio with LoRA."
    )
    # Data
    parser.add_argument("--base_model",      type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--data_root",       type=str, required=True)
    parser.add_argument("--transcripts_dir", type=str, default="_meta_full/train/transcripts")
    parser.add_argument("--output_dir",      type=str, default="runs/xlsr_lora_hineng")
    parser.add_argument("--eval_split",      type=float, default=0.10)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--strict",          action="store_true")
    # Duration filters  ← RAISED default to 4 hours
    parser.add_argument("--max_hours",   type=float, default=4.0,
                        help="Cap total audio. Default 4.0 h. Set None to use all.")
    parser.add_argument("--min_seg_sec", type=float, default=1.0)
    parser.add_argument("--max_seg_sec", type=float, default=20.0)
    # Training
    parser.add_argument("--num_train_epochs", type=int,   default=60)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--grad_accum",       type=int,   default=4)
    parser.add_argument("--learning_rate",    type=float, default=1e-4)   # ← lowered from 3e-4
    parser.add_argument("--warmup_ratio",     type=float, default=0.15)   # ← slightly longer
    # LoRA  ← FFN layers added to defaults
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,out_proj,intermediate_dense,output_dense",
        help="Comma-separated Linear layer name-suffixes to wrap with LoRA. "
             "Run with --list_lora_targets to see what is available.",
    )
    parser.add_argument("--list_lora_targets", action="store_true",
                        help="Print available Linear layer suffixes and exit.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ #
    # Optional: just list LoRA target candidates and exit                 #
    # ------------------------------------------------------------------ #
    if args.list_lora_targets:
        _model = Wav2Vec2ForCTC.from_pretrained(args.base_model)
        print_lora_target_candidates(_model)
        return

    # ------------------------------------------------------------------ #
    # 1. Load and filter Kaldi data                                        #
    # ------------------------------------------------------------------ #
    print(f"\n[data] Reading Kaldi files from "
          f"{os.path.join(args.data_root, args.transcripts_dir)}")
    entries = build_entries(args.data_root, args.transcripts_dir, strict=args.strict)
    print(f"[data] Loaded {len(entries)} utterances total")

    # Duration filter (only when segments present)
    if entries[0]["start"] is not None:
        before = len(entries)
        entries = [
            e for e in entries
            if args.min_seg_sec <= (e["end"] - e["start"]) <= args.max_seg_sec
        ]
        print(f"[data] Duration filter [{args.min_seg_sec}s, {args.max_seg_sec}s]: "
              f"{before} → {len(entries)} utterances")

    # Cap total hours — strategy: shuffle then accumulate until budget hit
    if args.max_hours is not None and entries[0]["start"] is not None:
        rng_cap   = np.random.default_rng(args.seed)
        idx_cap   = np.arange(len(entries))
        rng_cap.shuffle(idx_cap)
        budget_sec = args.max_hours * 3600.0
        kept, total_sec = [], 0.0
        for i in idx_cap:
            d = entries[i]["end"] - entries[i]["start"]
            if total_sec + d > budget_sec * 1.01:   # 1% slack
                continue
            kept.append(entries[i])
            total_sec += d
        entries = kept
        print(f"[data] Capped at {args.max_hours:.1f} h: "
              f"{len(entries)} utterances (~{total_sec/3600:.2f} h)")

    # Final report
    if entries[0]["start"] is not None:
        total_h = sum(e["end"] - e["start"] for e in entries) / 3600.0
        print(f"[data] Final dataset: {total_h:.2f} h — {len(entries)} utterances")

    if len(entries) < 50:
        raise RuntimeError(
            f"Only {len(entries)} utterances after filtering — too few to train. "
            "Check --max_hours, --min_seg_sec / --max_seg_sec, and missing audio files."
        )

    # ------------------------------------------------------------------ #
    # 2. Build vocab + processor                                           #
    # ------------------------------------------------------------------ #
    vocab_dict = build_vocab([e["text"] for e in entries])
    print(f"[vocab] Size: {len(vocab_dict)} characters")

    # Sanity check — these MUST be present
    for required in ("|", "[UNK]", "[PAD]"):
        assert required in vocab_dict, f"Vocab missing required token: {required!r}"

    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"[vocab] Saved to {vocab_path}")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16_000,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    processor.save_pretrained(args.output_dir)

    # Tokenizer decode sanity check
    sample_text = normalize_text(entries[0]["text"])
    sample_ids  = tokenizer(sample_text).input_ids
    sample_dec  = tokenizer.decode(sample_ids, skip_special_tokens=True)
    print(f"[vocab] Tokenizer round-trip check:")
    print(f"  input : {sample_text!r}")
    print(f"  ids   : {sample_ids[:10]}...")
    print(f"  decode: {sample_dec!r}")

    # ------------------------------------------------------------------ #
    # 3. Train / eval split                                                #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)
    cut          = int(len(entries) * (1.0 - args.eval_split))
    train_entries = [entries[i] for i in idx[:cut]]
    eval_entries  = [entries[i] for i in idx[cut:]]
    print(f"[split] Train: {len(train_entries)} | Eval: {len(eval_entries)}")

    train_ds = HinEngKaldiDataset(train_entries, processor)
    eval_ds  = HinEngKaldiDataset(eval_entries,  processor)

    # ------------------------------------------------------------------ #
    # 4. Base model                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n[model] Loading {args.base_model} ...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        # FIX: prevents -inf CTC loss from zeroing gradients in early epochs
        ctc_zero_infinity=True,
    )

    # Freeze CNN feature encoder (standard practice — it is already well-trained)
    model.freeze_feature_encoder()

    # FIX: re-initialize lm_head with small weights.
    # The default random init from ignore_mismatched_sizes can be large,
    # strongly biasing the softmax toward the blank token → blank collapse.
    with torch.no_grad():
        model.lm_head.weight.data.normal_(mean=0.0, std=0.02)
        model.lm_head.bias.data.zero_()
    print("[model] lm_head re-initialised (mean=0, std=0.02)")

    # ------------------------------------------------------------------ #
    # 5. LoRA wrapping                                                     #
    # ------------------------------------------------------------------ #
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    print(f"[lora] Target modules: {target_modules}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=None,
        # lm_head is NOT a LoRA target; put it in modules_to_save so PEFT
        # keeps it fully trainable alongside the adapter weights.
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # FIX: explicit gradient-flow verification —
    # confirm lm_head is trainable after PEFT wrapping
    lm_head_trainable = any(
        "lm_head" in n and p.requires_grad
        for n, p in model.named_parameters()
    )
    if not lm_head_trainable:
        # Force it — modules_to_save occasionally needs a nudge on some PEFT versions
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
                print(f"[lora] Manually enabled grad for {name}")

    verify_trainable_params(model)

    # ------------------------------------------------------------------ #
    # 6. Training arguments                                                #
    # ------------------------------------------------------------------ #
    # Effective batch = batch_size × grad_accum × n_gpus
    # Default: 8 × 4 = 32 per gradient step
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Batch / accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Schedule  ← cosine + longer warmup to escape blank collapse
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        # Eval / save
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        # Precision / perf
        bf16=True,
        gradient_checkpointing=False,
        group_by_length=True,
        dataloader_num_workers=2,

        # Prevent Trainer from stripping "input_values" / "labels" from batch
        remove_unused_columns=False,

        # Accumulate eval outputs in chunks to avoid OOM on longer eval sets
        eval_accumulation_steps=8,

        report_to="none",
    )

    # ------------------------------------------------------------------ #
    # 7. Trainer                                                           #
    # ------------------------------------------------------------------ #
    trainer = PeftAudioTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=make_compute_metrics(processor),
        # FIX: pass the full processor (not just feature_extractor)
        # so group_by_length and length-based sorting work correctly,
        # and so Trainer doesn't emit deprecation warnings
        processing_class=processor,
    )

    # ------------------------------------------------------------------ #
    # 8. Train                                                             #
    # ------------------------------------------------------------------ #
    print("\n[train] Starting training...")
    print(f"[train] Effective batch size: "
          f"{args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} per gradient step")
    trainer.train()

    # ------------------------------------------------------------------ #
    # 9. Save                                                              #
    # ------------------------------------------------------------------ #
    print(f"\n[save] Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Merge adapters into a standalone Wav2Vec2ForCTC for easy inference
    try:
        print("[save] Merging LoRA weights into base model ...")
        merged     = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"[save] Merged model saved to {merged_dir}")
    except Exception as exc:
        print(f"[save] merge_and_unload skipped: {exc}")

    print("\n[done] Training complete.")


if __name__ == "__main__":
    main()