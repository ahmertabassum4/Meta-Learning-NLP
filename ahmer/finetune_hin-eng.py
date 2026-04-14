import os
import json
import torch
import librosa
import numpy as np
import argparse
from typing import List, Dict, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)
from evaluate import load

def clean_text(text: str) -> str:
    # Remove markdown bold markers and other artifacts
    return text.replace("**", "").replace("__", "").strip()

class CSFleursDataset(Dataset):
    def __init__(self, metadata: List[Dict], base_path: str, processor: Wav2Vec2Processor):
        self.metadata = metadata
        self.base_path = base_path
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        # Joining base_path with the file_name from metadata
        # Handle the // in file_name observed in sample
        rel_path = item["file_name"].replace("//", "/")
        audio_path = os.path.join(self.base_path, rel_path)
        
        try:
            # Load audio using librosa to avoid AudioDecoder issues
            audio, _ = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Fallback to zeros if file missing or corrupt
            audio = np.zeros(16000)

        # Process audio
        input_values = self.processor(audio, sampling_rate=16000).input_values[0]
        
        # Process text
        text = clean_text(item["text"])
        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_values": input_values,
            "labels": labels
        }

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

def build_vocab(texts: List[str]) -> Dict[str, int]:
    all_chars = set()
    for text in texts:
        all_chars.update(clean_text(text))
    
    vocab = {c: i for i, c in enumerate(sorted(list(all_chars)))}
    
    # Add special tokens
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    # Map space to |
    if " " in vocab:
        vocab["|"] = vocab.pop(" ")
    else:
        vocab["|"] = len(vocab)
        
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--output_dir", type=str, default="runs/w2v2_csfleurs")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"], help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 1) Download dataset snapshot
    print("Downloading CS-FLEURS snapshot (Hindi-English subset)...")
    snapshot_path = snapshot_download(
        "byan/cs-fleurs",
        repo_type="dataset",
        allow_patterns=["xtts/train/metadata.jsonl", "xtts/train/audio/cs_hin_eng_n1_resample*"]
    )
    print(f"Snapshot downloaded to: {snapshot_path}")

    # 2) Load and filter metadata
    metadata_path = os.path.join(snapshot_path, "xtts/train/metadata.jsonl")
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("language") == "hin-eng":
                entries.append(obj)
    print("Num samples: ", len(entries))
    # Get the full duration in hours
    total_duration = sum([float(e["duration"]) for e in entries]) / 3600
    print(f"Total duration: {total_duration} hours")
    if args.max_samples:
        entries = entries[:args.max_samples]
    
    print(f"Total hin-eng samples: {len(entries)}")
    
    # 3) Setup Vocabulary and Processor
    vocab_dict = build_vocab([e["text"] for e in entries])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(args.output_dir, "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.output_dir)

    # 4) Create Datasets
    # Split into train/eval (90/10)
    np.random.seed(42)
    np.random.shuffle(entries)
    split_idx = int(len(entries) * 0.90)
    train_entries = entries[:split_idx]
    eval_entries = entries[split_idx:]

    base_audio_path = os.path.join(snapshot_path, "xtts/train")
    train_dataset = CSFleursDataset(train_entries, base_audio_path, processor)
    eval_dataset = CSFleursDataset(eval_entries, base_audio_path, processor)

    # 5) Setup Model
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_encoder()
    print('Model architecture: ', model)
    # 6) Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        save_steps=1000,
        eval_steps=500,
        logging_steps=10,
        # attention_dropout=0.2, # Add dropout to prevent overfitting
        # early_stopping=True, # Add early stopping
        # early_stopping_patience=5, # Stop training if eval loss doesn't improve for 5 epochs
        learning_rate=args.learning_rate,
        warmup_steps=500 ,
        save_total_limit=2,
        dataloader_num_workers=0, # Librosa is not thread-safe in some envs
        push_to_hub=False,
        report_to="none",
        # use_cpu = "mps" #use_cpu=(args.device == "cpu")
    )
    # wer_metric = load_metric("wer")
    # cer_metric = load_metric("cer")
    # 7) Trainer
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        args=training_args,
        compute_metrics= None, #[wer_metric, cer_metric], # Add WER if needed
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    
    # Save the final model and processor to output_dir
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

    """
    python ahmer/finetune_hin-eng.py --output_dir runs/w2v2_csfleurs --num_train_epochs 50 --batch_size 4  --learning_rate 3e-4  --device cuda 
    """