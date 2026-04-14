import os
import json
import torch
import librosa
import numpy as np
import argparse
import random
import io
import soundfile as sf
from typing import List, Dict, Union, Any
from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)
from evaluate import load as load_metric

# MONKEY PATCH: Disable torchcodec in datasets.features.audio to prevent the ImportError
import datasets.features.audio
def patched_decode_example(self, value, token_per_repo_id=None):
    if not self.decode:
        return value
    return value
datasets.features.audio.Audio.decode_example = patched_decode_example

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()

def build_vocab(texts: List[str]) -> Dict[str, int]:
    all_chars = set()
    for text in texts:
        all_chars.update(clean_text(text))
    vocab = {c: i for i, c in enumerate(sorted(list(all_chars)))}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    if " " in vocab:
        vocab["|"] = vocab.pop(" ")
    else:
        vocab["|"] = len(vocab)
    return vocab

def _get_duration_sec(audio_val: Dict) -> float:
    """Return duration of one audio sample (in seconds)."""
    audio_bytes = audio_val.get("bytes")
    if audio_bytes is not None:
        info = sf.info(io.BytesIO(audio_bytes))
        return info.duration
    audio_path = audio_val.get("path")
    audio, sr = librosa.load(audio_path, sr=None)
    return len(audio) / sr

def _standardise_hf_dataset(hf_ds, dataset_name: str):
    """Rename audio/text columns to canonical names and disable auto-decoding."""
    audio_col = next(
        (c for c in ["context", "audio", "correct_audio", "speech", "wav", "waveform"]
         if c in hf_ds.column_names),
        None,
    )
    if audio_col is None:
        raise ValueError(f"No audio column found in {dataset_name}. Columns: {hf_ds.column_names}")
    text_col = next(
        (c for c in ["text", "transcript", "correct_transcription", "transcription",
                      "sentence", "answer", "label"]
         if c in hf_ds.column_names),
        None,
    )
    if text_col is None:
        raise ValueError(f"No text column found in {dataset_name}. Columns: {hf_ds.column_names}")
    if audio_col != "audio":
        hf_ds = hf_ds.rename_column(audio_col, "audio")
    if text_col != "text":
        hf_ds = hf_ds.rename_column(text_col, "text")
    hf_ds = hf_ds.cast_column("audio", Audio(decode=False))
    return hf_ds

# ---------------------------------------------------------------------------
# Dataset & Collator
# ---------------------------------------------------------------------------

class SeameDataset(Dataset):
    def __init__(self, hf_dataset, processor: Wav2Vec2Processor):
        self.dataset = hf_dataset
        self.processor = processor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_val = item["audio"]
        audio_bytes = audio_val.get("bytes")
        if audio_bytes is not None:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # stereo → mono
        else:
            audio, sr = librosa.load(audio_val.get("path"), sr=16000)
        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        input_values = self.processor(audio, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(clean_text(item["text"])).input_ids
        return {"input_values": input_values, "labels": labels}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Direct Finetuning from Facebook XLS-R 300m")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Optional: Path to meta-learned processor to reuse vocabulary.")
    parser.add_argument("--output_dir", type=str, default="finetuned_1h_seame_direct")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=device)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load Target Data: SEAME Chinese-English (AudioLLMs/seame_dev_man)
    print("Loading AudioLLMs/seame_dev_man (test split) for 1-hour finetuning...")
    seame_raw = load_dataset("AudioLLMs/seame_dev_man", split="test")
    seame_hf = _standardise_hf_dataset(seame_raw, "SEAME_cmn_en")

    # Reproducible 1-hour subsetting with CTC validity check
    valid_indices, durations = [], []
    for i in range(len(seame_hf)):
        sample = seame_hf[i]
        try:
            dur = _get_duration_sec(sample["audio"])
            if (dur * 16000 // 320) >= len(clean_text(sample["text"])):
                valid_indices.append(i)
                durations.append(dur)
        except Exception:
            continue

    rng = random.Random(args.seed)
    combined = list(zip(valid_indices, durations))
    rng.shuffle(combined)
    train_indices, current_duration = [], 0
    for idx, dur in combined:
        if current_duration >= 1 * 3600:
            break
        train_indices.append(idx)
        current_duration += dur

    print(f"  Train subset : {len(train_indices)} samples ({current_duration/3600:.2f}h)")

    train_subset = seame_hf.select(train_indices)
    # Use up to 50 held-out samples for evaluation
    eval_indices = [c[0] for c in combined[len(train_indices):len(train_indices)+50]]
    eval_subset = seame_hf.select(eval_indices)
    print(f"  Eval  subset : {len(eval_indices)} samples")

    # 2) Setup Processor (Vocab)
    if args.model_path:
        print(f"Reusing processor from {args.model_path}")
        proc_path = os.path.join(args.model_path, "processor") if os.path.exists(os.path.join(args.model_path, "processor")) else args.model_path
        processor = Wav2Vec2Processor.from_pretrained(proc_path)
    else:
        print("Building new vocabulary from target data...")
        vocab_dict = build_vocab(train_subset["text"])
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "vocab.json"), "w") as f: json.dump(vocab_dict, f)
        tokenizer = Wav2Vec2CTCTokenizer(os.path.join(args.output_dir, "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(args.output_dir)

    # 3) Initialize Model from Facebook Checkpoint
    print("Initializing model from facebook/wav2vec2-xls-r-300m weights...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_encoder()
    model.to(device)

    # 4) Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=False,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        save_steps=args.eval_every,
        eval_steps=args.eval_every,
        logging_steps=10,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        report_to="none",
    )

    wer_metric, cer_metric = load_metric("wer"), load_metric("cer")
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str), 
                "cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=SeameDataset(train_subset, processor),
        eval_dataset=SeameDataset(eval_subset, processor),
    )

    print("Starting Direct Finetuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

# # To reuse vocab from a meta-learned model:
# python finetune_wav2vec2_seame_realcs_direct.py --model_path /path/to/metalearn_output

# # To start completely from scratch (new vocab from SEAME data):
# python finetune_wav2vec2_seame_realcs_direct.py --device cuda --output_dir finetuned_1h_seame_direct --num_train_epochs 10
