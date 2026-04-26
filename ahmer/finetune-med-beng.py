"""
Finetune Wav2Vec2 XLS-R on MediBeng (Bengali-English code-switched)
WITHOUT using the `datasets` library (avoids the torchcodec issue).

Instead:
  - Download MediBeng Parquet files via huggingface_hub.snapshot_download
  - Read rows with pandas
  - Decode embedded WAV bytes with librosa (via io.BytesIO)

Dataset: https://huggingface.co/datasets/pr0mila-gh0sh/MediBeng
  - train: 3,839 samples  (used for train + held-out validation)
  - test:  960 samples    (NOT touched during training — separate eval)

WARNING about MediBeng:
  The dataset only contains 24 unique text sentences. Both train and test
  have the same 24 sentences, just different TTS renditions. Do NOT report
  MediBeng-test WER as a generalization result — it's memorization.
  Use this model for transfer tests on CS-FLEURS and OpenSLR104 instead.

Usage:
  python ahmer/finetune_medibeng_hfhub.py \
    --output_dir runs/w2v2_medibeng_ben_eng \
    --num_train_epochs 15 \
    --batch_size 4 \
    --grad_accum 2 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.1 \
    --freeze_feature_encoder \
    --device cuda
"""

import os
import io
import re
import json
import random
import argparse
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Union

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from evaluate import load


# ---------------------------------------------------------------------------
# Text cleaning — must match CS-FLEURS script so vocabs/metrics are comparable
# ---------------------------------------------------------------------------
_KEEP_PATTERN = re.compile(
    r"[^"
    r"\u0980-\u09FF"   # Bengali block
    r"A-Za-z"
    r"0-9"
    r" "
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
# Dataset download + load
# ---------------------------------------------------------------------------
def download_medibeng(repo_id: str = "pr0mila-gh0sh/MediBeng",
                      cache_dir: str = None) -> str:
    """
    Download the MediBeng repo (only the Parquet data files) from HF Hub.
    Returns the local path to the downloaded snapshot.
    """
    print(f"Downloading {repo_id} from HuggingFace Hub...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        cache_dir=cache_dir,
    )
    print(f"Downloaded to: {local_dir}")
    return local_dir


def load_medibeng_split(parquet_path: str) -> pd.DataFrame:
    """
    Read a MediBeng Parquet file.

    The 'audio' column is a struct: {"bytes": <raw WAV bytes>, "path": str}.
    We keep it as-is and decode lazily in __getitem__.
    """
    print(f"Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class MediBengDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, processor: Wav2Vec2Processor,
                 audio_col: str = "audio", text_col: str = "text"):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.audio_col = audio_col
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # The "audio" cell is a dict-like: {"bytes": b"...", "path": "..."}
        audio_field = item[self.audio_col]
        audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) \
            else audio_field["bytes"]

        try:
            # librosa.load handles WAV bytes via io.BytesIO and resamples to 16kHz
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        except Exception as e:
            print(f"Error decoding audio at idx={idx}: {e}")
            audio = np.zeros(16000, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        text = clean_text(item[self.text_col])

        input_values = self.processor(
            audio, sampling_rate=16000
        ).input_values[0]

        labels = self.processor.tokenizer(
            text, add_special_tokens=False
        ).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
            "input_length": len(input_values),
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------
def build_vocab(texts: List[str]) -> Dict[str, int]:
    all_chars = set()
    for text in texts:
        all_chars.update(clean_text(text))
    all_chars.discard(" ")

    vocab_list = sorted(all_chars)
    vocab_dict = {c: i for i, c in enumerate(vocab_list)}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics_factory(processor: Wav2Vec2Processor):
    wer_metric = load("wer")
    cer_metric = load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        pred_str = [clean_text(x) if clean_text(x) else " " for x in pred_str]
        label_str = [clean_text(x) if clean_text(x) else " " for x in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def split_train_valid(df: pd.DataFrame, valid_ratio: float = 0.1,
                      seed: int = 42):
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)

    valid_size = max(1, int(len(df) * valid_ratio))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    return train_df, valid_df


def sanity_check_tokenizer(processor, sample_text):
    cleaned = clean_text(sample_text)
    ids = processor.tokenizer(cleaned, add_special_tokens=False).input_ids
    decoded = processor.tokenizer.decode(ids).replace(
        processor.tokenizer.pad_token, ""
    ).strip()
    print("\n[Tokenizer round-trip check]")
    print(f"  Original: {cleaned!r}")
    print(f"  Decoded : {decoded!r}")
    print(f"  Match   : {cleaned.replace(' ', '') == decoded.replace(' ', '')}")


def debug_model_predictions(model, processor, dataset, device, num_samples=3):
    model.eval()
    print("\n[Prediction debug]")
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            iv = torch.tensor(sample["input_values"]).unsqueeze(0).to(device)
            logits = model(iv).logits
            pred_ids = torch.argmax(logits, dim=-1)
            unique_ids = torch.unique(pred_ids).tolist()
            pred_text = processor.batch_decode(pred_ids)[0]
            label_ids = np.array(sample["labels"])
            ref_text = processor.tokenizer.decode(label_ids)
            print(f"  Sample {i}:")
            print(f"    Unique predicted token ids: {unique_ids}")
            print(f"    Pred: {pred_text[:150]!r}")
            print(f"    Ref : {ref_text[:150]!r}")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--hf_dataset", type=str,
                        default="pr0mila-gh0sh/MediBeng")
    parser.add_argument("--output_dir", type=str,
                        default="runs/w2v2_medibeng_ben_eng")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--freeze_feature_encoder", action="store_true",
                        default=True)
    parser.add_argument("--mask_time_prob", type=float, default=0.0)
    parser.add_argument("--layerdrop", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--hidden_dropout", type=float, default=0.0)
    parser.add_argument("--feat_proj_dropout", type=float, default=0.0)
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="If > 0, limit train split to first N samples")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF Hub cache dir (default: ~/.cache/huggingface)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Download + load
    # -----------------------------------------------------------------
    local_dir = download_medibeng(args.hf_dataset, cache_dir=args.cache_dir)

    # Find the Parquet files under data/
    data_dir = os.path.join(local_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected data/ dir under {local_dir}")

    parquet_files = sorted(f for f in os.listdir(data_dir)
                           if f.endswith(".parquet"))
    print(f"\nFound Parquet files: {parquet_files}")

    # Heuristic: files are named like train-00000-of-00001.parquet,
    # test-00000-of-00001.parquet
    train_files = [f for f in parquet_files if f.startswith("train")]
    test_files = [f for f in parquet_files if f.startswith("test")]
    if not train_files:
        raise FileNotFoundError(
            f"No train-*.parquet found in {data_dir}"
        )

    train_df_full = pd.concat(
        [load_medibeng_split(os.path.join(data_dir, f)) for f in train_files],
        ignore_index=True,
    )
    print(f"\nLoaded train rows (HF train split): {len(train_df_full)}")
    if test_files:
        print(f"HF test split present ({test_files[0]}) but NOT loaded "
              f"— run separate evaluation script for it.")

    if args.max_train_samples > 0:
        print(f"Limiting train split to first {args.max_train_samples} samples")
        train_df_full = train_df_full.head(args.max_train_samples).reset_index(
            drop=True
        )

    # Split HF train into train + validation (for in-training eval signal)
    train_df, valid_df = split_train_valid(
        train_df_full, valid_ratio=args.valid_ratio, seed=args.seed
    )

    print(f"\nTrain samples     : {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print("(HF test split is NOT used during training.)")

    # Save lightweight metadata (no audio) for reproducibility
    train_df[["text"]].to_csv(
        os.path.join(args.output_dir, "train_split.csv"), index=False
    )
    valid_df[["text"]].to_csv(
        os.path.join(args.output_dir, "valid_split.csv"), index=False
    )

    # -----------------------------------------------------------------
    # Peek at samples
    # -----------------------------------------------------------------
    print("\nSample training texts:")
    for t in train_df["text"].head(3).tolist():
        print("-", repr(clean_text(t)))
    print("\nSample validation texts:")
    for t in valid_df["text"].head(3).tolist():
        print("-", repr(clean_text(t)))

    # -----------------------------------------------------------------
    # Build vocab (from train + valid only; test is not loaded)
    # -----------------------------------------------------------------
    all_texts = (train_df["text"].astype(str).tolist()
                 + valid_df["text"].astype(str).tolist())
    vocab_dict = build_vocab(all_texts)
    print(f"\nVocab size (incl. special tokens): {len(vocab_dict)}")
    print(f"Vocab chars: "
          f"{sorted(c for c in vocab_dict if c not in ('[UNK]', '[PAD]', '|'))}")

    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------
    # Tokenizer + processor
    # -----------------------------------------------------------------
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
        do_lower_case=False,
    )
    tokenizer.bos_token = None
    tokenizer.eos_token = None
    tokenizer._bos_token = None
    tokenizer._eos_token = None

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    processor.save_pretrained(args.output_dir)

    sanity_check_tokenizer(processor, train_df["text"].iloc[0])

    train_dataset = MediBengDataset(train_df, processor)
    valid_dataset = MediBengDataset(valid_df, processor)

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=0.0,
        layerdrop=args.layerdrop,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = None
    model.config.eos_token_id = None

    # -----------------------------------------------------------------
    # CTC feasibility check
    # -----------------------------------------------------------------
    print("\n[CTC feasibility check]")
    problems = 0
    for i in range(min(len(train_dataset), 10)):
        sample = train_dataset[i]
        audio_len = len(sample["input_values"])
        label_len = len(sample["labels"])
        est_frames = audio_len // 320
        status = "OK" if est_frames >= label_len else "TOO SHORT"
        if status == "TOO SHORT":
            problems += 1
        print(f"  Sample {i}: audio=~{audio_len/16000:.2f}s, "
              f"frames~{est_frames}, label_len={label_len} [{status}]")

    # -----------------------------------------------------------------
    # Forward-pass sanity
    # -----------------------------------------------------------------
    print("\n[Direct forward-pass CTC loss check]")
    model.to(args.device)
    model.train()
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    _batch = collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])
    _batch = {k: v.to(args.device) for k, v in _batch.items()}
    with torch.no_grad():
        _out = model(**_batch)
    print(f"  Loss        : {_out.loss.item():.4f} "
          f"({'FINITE' if torch.isfinite(_out.loss) else 'NaN/Inf'})")
    print(f"  Logits shape: {tuple(_out.logits.shape)}")
    print(f"  Vocab size  : {len(processor.tokenizer)}")
    print(f"  Pad token id: {processor.tokenizer.pad_token_id}")
    print(f"  Tokenizer bos/eos: "
          f"{processor.tokenizer.bos_token_id}/{processor.tokenizer.eos_token_id}")
    del _batch, _out

    use_fp16 = (args.device == "cuda")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        length_column_name="input_length",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        dataloader_num_workers=0,
        fp16=use_fp16,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=processor,
        callbacks=callbacks,
        compute_metrics=compute_metrics_factory(processor),
    )

    device = next(model.parameters()).device
    debug_model_predictions(model, processor, valid_dataset, device, num_samples=2)

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    debug_model_predictions(model, processor, valid_dataset, device, num_samples=3)

    print("\nRunning final evaluation on VALIDATION split "
          "(HF test split is reserved for separate eval)...")
    val_metrics = trainer.evaluate(
        eval_dataset=valid_dataset, metric_key_prefix="final_val"
    )
    with open(os.path.join(args.output_dir, "final_val_metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    print("Final validation metrics:")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))

    print(f"\nTraining complete. Model saved at: {args.output_dir}")
    print("NOTE: HF test split (960 samples) was NOT evaluated here.")
    print("For honest generalization numbers, evaluate on CS-FLEURS test "
          "and OpenSLR104 instead (MediBeng test ≈ memorization check only).")


if __name__ == "__main__":
    main()