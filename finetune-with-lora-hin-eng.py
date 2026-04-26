"""
Finetune facebook/wav2vec2-xls-r-300m on Hindi-English code-switched audio using LoRA.

Expected data layout (Kaldi-style, matches your screenshot):

    <data_root>/
        _meta_full/train/transcripts/
            text         # <utt_id> <transcription>
            wav.scp      # <recording_id> <path/to/wav>
            segments     # <utt_id> <recording_id> <start_sec> <end_sec>   (optional)
            utt2spk      # <utt_id> <spk_id>                               (unused here)
        audio/
            *.wav

If `segments` is present, audio is cropped per utterance. Otherwise each utt_id
is assumed to map 1:1 to a wav (utt_id == recording_id in wav.scp).

WER and CER are computed on the eval set every `eval_steps`.

Example:
    python finetune_xlsr_lora_hineng.py \
        --data_root /workspace/projects/asr_project_src/hineng_cs_1hr \
        --transcripts_dir _meta_full/train/transcripts \
        --output_dir runs/xlsr_lora_hineng_1hr \
        --num_train_epochs 30 \
        --batch_size 8 \
        --grad_accum 2 \
        --learning_rate 3e-4
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
from peft import LoraConfig, get_peft_model, TaskType
import evaluate


# ----------------------------------------------------------------------------- #
# Kaldi file parsing                                                            #
# ----------------------------------------------------------------------------- #

def read_kaldi_two_col(path: str) -> Dict[str, str]:
    """Read a Kaldi file where each line is `<key> <rest_of_line>`."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                out[parts[0]] = ""
            else:
                out[parts[0]] = parts[1]
    return out


def read_segments(path: str) -> Dict[str, Dict]:
    """segments: <utt_id> <rec_id> <start_sec> <end_sec>"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            utt_id, rec_id, start, end = parts[0], parts[1], float(parts[2]), float(parts[3])
            out[utt_id] = {"rec_id": rec_id, "start": start, "end": end}
    return out


def _index_audio_dir(audio_dir: str) -> Dict[str, str]:
    """Map every basename (with and without extension) under audio_dir to its
    full path. Used to recover when wav.scp paths point to a stale location."""
    index = {}
    if not os.path.isdir(audio_dir):
        return index
    for root, _, files in os.walk(audio_dir):
        for fn in files:
            full = os.path.join(root, fn)
            index[fn] = full                      # e.g. "abc123.wav"
            stem, _ = os.path.splitext(fn)
            index.setdefault(stem, full)          # e.g. "abc123"
    return index


def resolve_wav_path(scp_value: str, data_root: str, audio_index: Dict[str, str]) -> Union[str, None]:
    """Resolve a wav.scp value to an existing file on disk, or return None.

    Handles three cases:
      1. Plain absolute path            -> use as-is if it exists
      2. Plain relative path            -> try data_root/<rel>
      3. Kaldi pipe entry (ends in '|') -> not a file we can librosa.load
      Falls back to looking up the basename in the local audio/ directory.
    """
    val = scp_value.strip()
    if not val:
        return None

    # Kaldi pipe entry — would need to be executed, not a file path.
    # We don't support these; signal failure so the entry is dropped.
    if val.endswith("|"):
        # Try to recover a basename from the pipe command anyway
        for tok in val.split():
            if tok.endswith(".wav") or tok.endswith(".flac"):
                hit = audio_index.get(os.path.basename(tok))
                if hit:
                    return hit
        return None

    # Direct path checks
    if os.path.isabs(val) and os.path.exists(val):
        return val
    cand = os.path.join(data_root, val)
    if os.path.exists(cand):
        return cand

    # Fall back to basename lookup in the local audio/ index
    base = os.path.basename(val)
    return audio_index.get(base) or audio_index.get(os.path.splitext(base)[0])


def build_entries(data_root: str, transcripts_dir: str, strict: bool = False) -> List[Dict]:
    """Produce a list of dicts: {utt_id, text, wav_path, start, end}.

    Drops utterances whose audio cannot be located, and prints a summary.
    """
    tdir = os.path.join(data_root, transcripts_dir)
    text = read_kaldi_two_col(os.path.join(tdir, "text"))
    wav_scp = read_kaldi_two_col(os.path.join(tdir, "wav.scp"))

    segments_path = os.path.join(tdir, "segments")
    has_segments = os.path.exists(segments_path)
    segments = read_segments(segments_path) if has_segments else {}

    audio_index = _index_audio_dir(os.path.join(data_root, "audio"))
    print(f"Indexed {len(audio_index)} audio file basenames under {data_root}/audio")

    entries = []
    missing_rec_in_scp = 0
    missing_seg = 0
    unresolved_path = 0
    bad_examples: List[str] = []

    for utt_id, transcription in text.items():
        if has_segments:
            seg = segments.get(utt_id)
            if seg is None:
                missing_seg += 1
                continue
            rec_id = seg["rec_id"]
            start, end = seg["start"], seg["end"]
        else:
            rec_id = utt_id
            start, end = None, None

        wav_val = wav_scp.get(rec_id)
        if wav_val is None:
            missing_rec_in_scp += 1
            continue

        wav_path = resolve_wav_path(wav_val, data_root, audio_index)
        if wav_path is None:
            unresolved_path += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"  utt={utt_id}  rec={rec_id}  scp_value={wav_val!r}")
            continue

        entries.append({
            "utt_id": utt_id,
            "text": transcription,
            "wav_path": wav_path,
            "start": start,
            "end": end,
        })

    total_utts = len(text)
    dropped = missing_seg + missing_rec_in_scp + unresolved_path
    print(
        f"Resolved {len(entries)}/{total_utts} utterances "
        f"(dropped: missing_segments={missing_seg}, "
        f"rec_id_not_in_wav_scp={missing_rec_in_scp}, "
        f"file_not_found={unresolved_path})"
    )
    if bad_examples:
        print("Examples of unresolved paths:")
        for ex in bad_examples:
            print(ex)

    if strict and dropped > 0:
        raise RuntimeError(
            f"--strict was set and {dropped} utterances could not be resolved. "
            f"Inspect the wav.scp / audio directory and try again."
        )
    if len(entries) == 0:
        raise RuntimeError(
            "Zero usable utterances. Check that wav.scp recording IDs match "
            "filenames in your audio/ directory."
        )
    return entries


# ----------------------------------------------------------------------------- #
# Text + vocab                                                                  #
# ----------------------------------------------------------------------------- #

def clean_text(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def build_vocab(texts: List[str]) -> Dict[str, int]:
    all_chars = set()
    for t in texts:
        # Treat space as | from the start
        all_chars.update(clean_text(t).replace(" ", "|"))

    # Remove raw space if it crept in
    all_chars.discard(" ")
    
    vocab = {c: i for i, c in enumerate(sorted(all_chars))}
    vocab.setdefault("|", len(vocab))      # word delimiter — must exist
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


# ----------------------------------------------------------------------------- #
# Dataset + collator                                                            #
# ----------------------------------------------------------------------------- #

class HinEngKaldiDataset(Dataset):
    def __init__(self, entries: List[Dict], processor: Wav2Vec2Processor, sr: int = 16000):
        self.entries = entries
        self.processor = processor
        self.sr = sr

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        try:
            if e["start"] is not None:
                offset = e["start"]
                duration = e["end"] - e["start"]
                audio, _ = librosa.load(
                    e["wav_path"], sr=self.sr, offset=offset, duration=duration
                )
            else:
                audio, _ = librosa.load(e["wav_path"], sr=self.sr)
        except Exception as err:
            # Upfront validation in build_entries() should catch missing files,
            # so a runtime failure here is unexpected — surface it loudly.
            raise RuntimeError(
                f"Failed to load {e['wav_path']} for utt {e['utt_id']}: {err}"
            ) from err

        # input_values = self.processor(audio, sampling_rate=self.sr).input_values[0]
        input_values = self.processor.feature_extractor(audio, sampling_rate=self.sr).input_values[0]
        text = clean_text(e["text"]).replace(" ", "|")
        labels = self.processor.tokenizer(text).input_ids
        return {"input_values": input_values, "labels": labels}

# class PeftAudioTrainer(Trainer):
#     """Trainer that saves PEFT adapters without touching nonexistent token embeddings."""
#     def _save(self, output_dir=None, state_dict=None):
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         # Wav2Vec2 has no input token embeddings; tell PEFT not to look for them.
#         self.model.save_pretrained(output_dir, save_embedding_layers=False)
#         if self.processing_class is not None:
#             self.processing_class.save_pretrained(output_dir)
#         torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class PeftAudioTrainer(Trainer):
    """Trainer that (1) saves PEFT adapters without touching nonexistent token
    embeddings, and (2) runs a correct prediction_step for PEFT-wrapped audio
    CTC models so compute_metrics actually receives logits."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run a forward pass and return (loss, logits, labels).

        We have to write this ourselves because Trainer's default
        prediction_step inspects the model's `forward` signature to decide
        which keys to pass — and PeftModel.forward(*args, **kwargs) reports
        no named args, so Trainer ends up passing nothing useful.
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(
                    input_values=inputs["input_values"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                )
            loss = outputs.loss.detach() if has_labels else None
            logits = outputs.logits.detach()
            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.detach()

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

# @dataclass
# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_values": f["input_values"]} for f in features]
#         label_features = [{"input_ids": f["labels"]} for f in features]

#         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
#         labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch["labels"] = labels
#         return batch
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Audio: route through feature_extractor.pad (returns "input_values")
        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        # Labels: route through tokenizer.pad (returns "input_ids" -> we rename to "labels")
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

# ----------------------------------------------------------------------------- #
# Metrics: WER + CER                                                            #
# ----------------------------------------------------------------------------- #


def make_compute_metrics(processor: Wav2Vec2Processor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = np.asarray(pred.predictions, dtype=np.float32)
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = np.array(pred.label_ids, copy=True)
        # Replace -100 with pad_token_id so tokenizer can decode
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # FIX 1: skip_special_tokens=True removes [PAD] and [UNK] from both strings
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_str  = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Debug: print a few pairs to visually inspect
        for p, r in zip(pred_str[:3], ref_str[:3]):
            print(f"  REF : {r!r}")
            print(f"  PRED: {p!r}")

        pairs = [(p, r) for p, r in zip(pred_str, ref_str) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        preds, refs = zip(*pairs)

        wer = wer_metric.compute(predictions=list(preds), references=list(refs))
        cer = cer_metric.compute(predictions=list(preds), references=list(refs))
        return {"wer": round(wer, 4), "cer": round(cer, 4)}

    return compute_metrics


# ----------------------------------------------------------------------------- #
# Main                                                                          #
# ----------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root containing _meta_full/ and audio/")
    parser.add_argument("--transcripts_dir", type=str, default="_meta_full/train/transcripts",
                        help="Path under data_root holding text/wav.scp/segments")
    parser.add_argument("--output_dir", type=str, default="runs/xlsr_lora_hineng")
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--eval_split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,out_proj",
                        help="Comma-separated module name suffixes to wrap with LoRA")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any utterance audio cannot be located")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="Cap total training+eval audio at roughly this many hours "
                             "(samples shorter segments first for stable LoRA training)")
    parser.add_argument("--min_seg_sec", type=float, default=1.0,
                        help="Drop segments shorter than this (CTC needs enough frames)")
    parser.add_argument("--max_seg_sec", type=float, default=20.0,
                        help="Drop segments longer than this (memory + bad alignments)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load Kaldi data
    print(f"Reading Kaldi files from {os.path.join(args.data_root, args.transcripts_dir)}")
    entries = build_entries(args.data_root, args.transcripts_dir, strict=args.strict)
    print(f"Loaded {len(entries)} utterances")
    if len(entries) == 0:
        raise RuntimeError("No utterances found — check --data_root and --transcripts_dir")

    # 1a) Filter by segment duration (only meaningful when we have segments)
    if entries[0]["start"] is not None:
        before = len(entries)
        entries = [
            e for e in entries
            if args.min_seg_sec <= (e["end"] - e["start"]) <= args.max_seg_sec
        ]
        print(f"Duration filter [{args.min_seg_sec:.1f}s, {args.max_seg_sec:.1f}s]: "
              f"{before} -> {len(entries)} utterances")

    # 1b) Optional cap on total hours.
    # Strategy: shuffle deterministically, then take entries until we hit the cap.
    # We don't sort by length — that would bias the model toward short utts.
    if args.max_hours is not None and entries[0]["start"] is not None:
        rng_cap = np.random.default_rng(args.seed)
        idx_cap = np.arange(len(entries))
        rng_cap.shuffle(idx_cap)
        budget_sec = args.max_hours * 3600.0
        kept, total = [], 0.0
        for i in idx_cap:
            d = entries[i]["end"] - entries[i]["start"]
            if total + d > budget_sec:
                continue
            kept.append(entries[i])
            total += d
            if total >= budget_sec:
                break
        entries = kept
        print(f"Capped at --max_hours={args.max_hours}: kept {len(entries)} utts "
              f"(~{total/3600:.2f} h)")

    # Final duration report
    if entries[0]["start"] is not None:
        total_h = sum(e["end"] - e["start"] for e in entries) / 3600.0
        print(f"Final dataset duration: {total_h:.2f} h ({len(entries)} utterances)")

    # 2) Vocab + processor
    vocab_dict = build_vocab([e["text"] for e in entries])
    print(f"Vocab size: {len(vocab_dict)} chars")
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.output_dir)

    # 3) Train / eval split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)
    cut = int(len(entries) * (1.0 - args.eval_split))
    train_entries = [entries[i] for i in idx[:cut]]
    eval_entries = [entries[i] for i in idx[cut:]]
    print(f"Train: {len(train_entries)} | Eval: {len(eval_entries)}")

    train_ds = HinEngKaldiDataset(train_entries, processor)
    eval_ds = HinEngKaldiDataset(eval_entries, processor)

    # 4) Base model
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    # 5) Wrap with LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    # lora_cfg = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     target_modules=target_modules,
    #     # FEATURE_EXTRACTION keeps the CTC head trainable as a "modules_to_save"
    #     task_type=TaskType.FEATURE_EXTRACTION,
    #     modules_to_save=["lm_head"],
    # )
    lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    target_modules=target_modules,
    task_type=None,                  # ← changed from TaskType.FEATURE_EXTRACTION
    modules_to_save=["lm_head"],     # ← still works; CTC head stays trainable
)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 6) Training args (RTX 5090 — 32 GB, bf16 supported)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=20,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
        # prediction_loss_only=False,        # <-- ADD THIS
        # include_inputs_for_metrics=False,  # <-- ADD THIS (defensive)
        # eval_accumulation_steps=4, 
    )

    trainer = PeftAudioTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=make_compute_metrics(processor),
    processing_class=processor,          # ← pass full processor, not .feature_extractor
)

    print("Starting training...")
    trainer.train()

    # 7) Save LoRA adapter + processor
    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Optional: also save a merged full model for easy inference later
    try:
        merged = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged")
        # The merged model is plain Wav2Vec2ForCTC again — no PEFT save flags needed.
        merged.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")
    except Exception as e:
        print(f"[note] merge_and_unload skipped: {e}")


if __name__ == "__main__":
    main()