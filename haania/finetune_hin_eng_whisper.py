import os
import json
import torch
import librosa
import numpy as np
import argparse
from typing import List, Dict, Union, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from evaluate import load


def clean_text(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


class CSFleursWhisperDataset(Dataset):
    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        base_path: str,
        processor: WhisperProcessor,
        max_audio_seconds: Optional[float] = None,
    ):
        self.metadata = metadata
        self.base_path = base_path
        self.processor = processor
        self.max_audio_seconds = max_audio_seconds

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        rel_path = item["file_name"].replace("//", "/")
        audio_path = os.path.join(self.base_path, rel_path)

        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(16000, dtype=np.float32)

        if self.max_audio_seconds is not None:
            max_len = int(16000 * self.max_audio_seconds)
            audio = audio[:max_len]

        text = clean_text(item["text"])

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


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

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [clean_text(x) for x in pred_str]
        label_str = [clean_text(x) for x in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


def infer_train_audio_pattern(language_pair: str) -> str:
    return f"xtts/train/audio/cs_{language_pair.replace('-', '_')}_n1_resample/*"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--output_dir", type=str, default="runs/whisper_csfleurs_hin_eng")
    parser.add_argument("--language_pair", type=str, default="hin-eng")
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

    print(f"Using device: {args.device}")
    print(f"Downloading CS-FLEURS snapshot ({args.language_pair})...")

    snapshot_path = snapshot_download(
        "byan/cs-fleurs",
        repo_type="dataset",
        allow_patterns=[
            "xtts/train/metadata.jsonl",
            infer_train_audio_pattern(args.language_pair),
        ],
    )
    print(f"Snapshot downloaded to: {snapshot_path}")

    metadata_path = os.path.join(snapshot_path, "xtts/train/metadata.jsonl")
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("language") == args.language_pair:
                entries.append(obj)

    print("Num samples:", len(entries))
    total_duration = sum(float(e["duration"]) for e in entries) / 3600
    print(f"Total duration: {total_duration:.2f} hours")

    if args.max_samples:
        entries = entries[:args.max_samples]

    print(f"Total selected samples: {len(entries)}")

    processor = WhisperProcessor.from_pretrained(args.base_model)

    np.random.seed(42)
    np.random.shuffle(entries)
    split_idx = int(len(entries) * 0.90)
    train_entries = entries[:split_idx]
    eval_entries = entries[split_idx:]

    base_audio_path = os.path.join(snapshot_path, "xtts/train")
    train_dataset = CSFleursWhisperDataset(
        train_entries,
        base_audio_path,
        processor,
        max_audio_seconds=args.max_audio_seconds,
    )
    eval_dataset = CSFleursWhisperDataset(
        eval_entries,
        base_audio_path,
        processor,
        max_audio_seconds=args.max_audio_seconds,
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # Better default for code-switching / multilingual ASR
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Model architecture:", model.__class__.__name__)

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
        eval_steps=500,
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
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()