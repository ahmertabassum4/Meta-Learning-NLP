import os
import csv
import torch
import librosa
import numpy as np
import argparse
from typing import List, Dict, Union, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from evaluate import load


import re
import string

def clean_text(text: str) -> str:
    text = text.replace("**", "").replace("__", "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_manifest_csv(manifest_path: str) -> List[Dict[str, Any]]:
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def resolve_audio_path(split_dir: str, row: Dict[str, Any]) -> str:
    for key in ["audio", "audio_path", "file_name", "path", "wav", "filename"]:
        if key in row and row[key]:
            rel = row[key].strip()
            return rel if os.path.isabs(rel) else os.path.join(split_dir, rel)
    raise KeyError(
        "Could not find audio path column in manifest.csv. "
        "Expected one of: audio, audio_path, file_name, path, wav, filename"
    )


def resolve_text(row: Dict[str, Any]) -> str:
    for key in ["text", "transcript", "sentence", "normalized_text"]:
        if key in row and row[key]:
            return clean_text(row[key])
    raise KeyError(
        "Could not find transcript column in manifest.csv. "
        "Expected one of: text, transcript, sentence, normalized_text"
    )


class RepoWhisperDataset(Dataset):
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        split_dir: str,
        processor: WhisperProcessor,
        max_audio_seconds: Optional[float] = None,
    ):
        self.entries = entries
        self.split_dir = split_dir
        self.processor = processor
        self.max_audio_seconds = max_audio_seconds

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        audio_path = resolve_audio_path(self.split_dir, item)

        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(16000, dtype=np.float32)

        if self.max_audio_seconds is not None:
            max_len = int(16000 * self.max_audio_seconds)
            audio = audio[:max_len]

        text = resolve_text(item)

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
def filter_long_examples(entries, processor, max_label_length=448):
    kept = []
    dropped = 0

    for item in entries:
        text = resolve_text(item)
        ids = processor.tokenizer(
            text,
            add_special_tokens=True,
        ).input_ids

        if len(ids) <= max_label_length:
            kept.append(item)
        else:
            dropped += 1

    print(f"Dropped {dropped} samples with label length > {max_label_length}")
    return kept

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
    parser.add_argument("--output_dir", type=str, default="runs/whisper_repo_ben_eng")
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path like data/cs_fleurs_ben_eng_300_49/train",
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_audio_seconds", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    manifest_path = os.path.join(args.train_dir, "manifest.csv")

    print(f"Using device: {args.device}")
    print(f"Loading training data from: {args.train_dir}")

    entries = read_manifest_csv(manifest_path)

    if args.max_samples is not None:
        entries = entries[:args.max_samples]

    print(f"Total selected samples: {len(entries)}")

    processor = WhisperProcessor.from_pretrained(args.base_model)
    
    processor = WhisperProcessor.from_pretrained(args.base_model)
    entries = filter_long_examples(entries, processor, max_label_length=448)
    print(f"Total usable samples after filtering: {len(entries)}")

    np.random.seed(42)
    np.random.shuffle(entries)
    split_idx = int(len(entries) * 0.90)
    train_entries = entries[:split_idx]
    eval_entries = entries[split_idx:]

    print(f"Train samples: {len(train_entries)}")
    print(f"Eval samples: {len(eval_entries)}")

    train_dataset = RepoWhisperDataset(
        train_entries,
        args.train_dir,
        processor,
        max_audio_seconds=args.max_audio_seconds,
    )
    eval_dataset = RepoWhisperDataset(
        eval_entries,
        args.train_dir,
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
        save_strategy="steps",
        save_steps=1000,
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        fp16=(args.device == "cuda"),
        push_to_hub=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics_factory(processor),
        processing_class=processor,
    )

    print("Starting training...")
    train_results = trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
   
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()