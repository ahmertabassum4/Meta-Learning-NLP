import os
import io
import re
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from evaluate import load


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("**", "").replace("__", "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def download_medibeng(repo_id: str = "pr0mila-gh0sh/MediBeng", cache_dir: str = None) -> str:
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
    print(f"Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    return df


def split_train_valid(df: pd.DataFrame, valid_ratio: float = 0.1, seed: int = 42):
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)

    valid_size = max(1, int(len(df) * valid_ratio))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    return train_df, valid_df


class MediBengWhisperDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor: WhisperProcessor,
        audio_col: str = "audio",
        text_col: str = "text",
        max_audio_seconds: Optional[float] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.audio_col = audio_col
        self.text_col = text_col
        self.max_audio_seconds = max_audio_seconds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        audio_field = item[self.audio_col]
        audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field["bytes"]

        try:
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            print(f"Error decoding audio at idx={idx}: {e}")
            audio = np.zeros(16000, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        if self.max_audio_seconds is not None:
            max_len = int(16000 * self.max_audio_seconds)
            audio = audio[:max_len]

        text = clean_text(item[self.text_col])

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=448,
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


def filter_long_examples(df: pd.DataFrame, processor: WhisperProcessor, text_col: str = "text", max_label_length: int = 448):
    keep_rows = []
    dropped = 0

    for i in range(len(df)):
        text = clean_text(df.iloc[i][text_col])
        ids = processor.tokenizer(
            text,
            add_special_tokens=True,
        ).input_ids

        if len(ids) <= max_label_length:
            keep_rows.append(i)
        else:
            dropped += 1

    filtered_df = df.iloc[keep_rows].reset_index(drop=True)
    print(f"Dropped {dropped} samples with label length > {max_label_length}")
    return filtered_df


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics_factory(processor: WhisperProcessor):
    wer_metric = load("wer")
    cer_metric = load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [clean_text(x) for x in pred_str]
        label_str = [clean_text(x) for x in label_str]

        print("\nSample predictions during eval:")
        for i in range(min(10, len(pred_str))):
            print(f"\n--- Example {i+1} ---")
            print("REF :", label_str[i])
            print("PRED:", pred_str[i])

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--hf_dataset", type=str, default="pr0mila-gh0sh/MediBeng")
    parser.add_argument("--output_dir", type=str, default="runs/whisper_medibeng_ben_eng")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_audio_seconds", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    local_dir = download_medibeng(args.hf_dataset, cache_dir=args.cache_dir)

    data_dir = os.path.join(local_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected data/ dir under {local_dir}")

    parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    print(f"\nFound Parquet files: {parquet_files}")

    train_files = [f for f in parquet_files if f.startswith("train")]
    test_files = [f for f in parquet_files if f.startswith("test")]
    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet found in {data_dir}")

    train_df_full = pd.concat(
        [load_medibeng_split(os.path.join(data_dir, f)) for f in train_files],
        ignore_index=True,
    )
    print(f"\nLoaded train rows (HF train split): {len(train_df_full)}")
    if test_files:
        print(f"HF test split present ({test_files[0]}) but NOT loaded — reserved for separate eval.")

    if args.max_train_samples > 0:
        print(f"Limiting train split to first {args.max_train_samples} samples")
        train_df_full = train_df_full.head(args.max_train_samples).reset_index(drop=True)

    processor = WhisperProcessor.from_pretrained(args.base_model)

    train_df_full = filter_long_examples(train_df_full, processor, text_col="text", max_label_length=448)
    print(f"Usable train rows after filtering: {len(train_df_full)}")

    train_df, valid_df = split_train_valid(
        train_df_full, valid_ratio=args.valid_ratio, seed=args.seed
    )

    print(f"\nTrain samples     : {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print("(HF test split is NOT used during training.)")

    train_df[["text"]].to_csv(
        os.path.join(args.output_dir, "train_split.csv"), index=False
    )
    valid_df[["text"]].to_csv(
        os.path.join(args.output_dir, "valid_split.csv"), index=False
    )

    print("\nSample training texts:")
    for t in train_df["text"].head(3).tolist():
        print("-", repr(clean_text(t)))

    print("\nSample validation texts:")
    for t in valid_df["text"].head(3).tolist():
        print("-", repr(clean_text(t)))

    train_dataset = MediBengWhisperDataset(
        train_df,
        processor,
        max_audio_seconds=args.max_audio_seconds,
    )
    valid_dataset = MediBengWhisperDataset(
        valid_df,
        processor,
        max_audio_seconds=args.max_audio_seconds,
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"
    model.generation_config.language = "bn"

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        fp16=(args.device == "cuda"),
        push_to_hub=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics_factory(processor),
        processing_class=processor,
        callbacks=callbacks,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    final_val_metrics = trainer.evaluate(
        eval_dataset=valid_dataset, metric_key_prefix="final_val"
    )
    with open(os.path.join(args.output_dir, "final_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_val_metrics, f, indent=2, ensure_ascii=False)

    print("\nFinal validation metrics:")
    print(json.dumps(final_val_metrics, indent=2, ensure_ascii=False))

    print(f"\nTraining complete. Model saved at: {args.output_dir}")
    print("NOTE: HF test split was NOT evaluated here.")
    print("For honest generalization numbers, evaluate on CS-FLEURS test and OpenSLR104.")


if __name__ == "__main__":
    main()