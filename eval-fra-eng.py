#!/usr/bin/env python
"""
Evaluate a Wav2Vec2-CTC LoRA model on held-out test data.

Works with:
1) a PEFT adapter directory, e.g. finetuned_lora_fra/
2) a merged full model directory, e.g. finetuned_lora_fra/merged/

Example:
    # Evaluate merged model
    python evaluate_fra_eng.py \
        --model_path finetuned_lora_fra/merged \
        --test_manifest finetuned_lora_fra/test_pool.json \
        --device cuda

    # Evaluate LoRA adapter + base model
    python evaluate_fra_eng.py \
        --model_path finetuned_lora_fra \
        --base_model facebook/wav2vec2-xls-r-300m \
        --test_manifest finetuned_lora_fra/test_pool.json \
        --device cuda

Output:
    <output_dir>/eval_results.json
    <output_dir>/predictions.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except Exception:
    EVALUATE_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False


import warnings
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa.*")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def choose_device(user_device: Optional[str] = None) -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_text(t: str) -> str:
    if t is None:
        return ""
    return str(t).strip().lower()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_cer_simple(refs: List[str], hyps: List[str]) -> float:
    def edit_distance(a: List[str], b: List[str]) -> int:
        dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
        dp[:, 0] = np.arange(len(a) + 1)
        dp[0, :] = np.arange(len(b) + 1)
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,
                    dp[i, j - 1] + 1,
                    dp[i - 1, j - 1] + cost,
                )
        return int(dp[len(a), len(b)])

    total_dist = 0
    total_chars = 0
    for r, h in zip(refs, hyps):
        r_chars = list(r)
        h_chars = list(h)
        total_dist += edit_distance(r_chars, h_chars)
        total_chars += max(1, len(r_chars))
    return total_dist / total_chars


def compute_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    refs = [clean_text(x) for x in refs]
    hyps = [clean_text(x) for x in hyps]

    if EVALUATE_AVAILABLE:
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        wer = wer_metric.compute(predictions=hyps, references=refs)
        cer = cer_metric.compute(predictions=hyps, references=refs)
        return {"wer": float(wer), "cer": float(cer)}

    if JIWER_AVAILABLE:
        wer = jiwer.wer(refs, hyps)
        try:
            cer = jiwer.cer(refs, hyps)
        except Exception:
            cer = compute_cer_simple(refs, hyps)
        return {"wer": float(wer), "cer": float(cer)}

    # fallback without external metrics libs
    def word_edit_distance(a: List[str], b: List[str]) -> int:
        dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
        dp[:, 0] = np.arange(len(a) + 1)
        dp[0, :] = np.arange(len(b) + 1)
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,
                    dp[i, j - 1] + 1,
                    dp[i - 1, j - 1] + cost,
                )
        return int(dp[len(a), len(b)])

    total_word_dist = 0
    total_words = 0
    for r, h in zip(refs, hyps):
        rw = r.split()
        hw = h.split()
        total_word_dist += word_edit_distance(rw, hw)
        total_words += max(1, len(rw))

    wer = total_word_dist / total_words
    cer = compute_cer_simple(refs, hyps)
    return {"wer": float(wer), "cer": float(cer)}


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class AudioManifestDataset(Dataset):
    def __init__(self, manifest: List[Dict[str, Any]], processor: Wav2Vec2Processor):
        self.items = manifest
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio_path = item["audio_path"]
        start = float(item.get("start_time", 0.0))
        end = item.get("end_time", None)
        duration = (float(end) - start) if end is not None else None

        try:
            audio, _ = librosa.load(
                audio_path,
                sr=16000,
                offset=start,
                duration=duration,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {audio_path} | {e}") from e

        input_values = self.processor(audio, sampling_rate=16000).input_values[0]
        text = clean_text(item.get("text", ""))

        return {
            "input_values": input_values,
            "reference": text,
            "audio_path": audio_path,
        }


@dataclass
class EvalCollator:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["references"] = [f["reference"] for f in features]
        batch["audio_paths"] = [f["audio_path"] for f in features]
        return batch


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def is_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def load_model_and_processor(model_path: str, base_model: Optional[str], device: str):
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    if is_adapter_dir(model_path):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "This model path looks like a PEFT adapter, but peft is not installed."
            )
        if not base_model:
            raise ValueError(
                "--base_model is required when --model_path points to a LoRA adapter directory."
            )

        base = Wav2Vec2ForCTC.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return processor, model


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    processor,
    manifest_path: str,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    dataset = AudioManifestDataset(manifest, processor)
    collator = EvalCollator(processor=processor, padding=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    all_refs: List[str] = []
    all_hyps: List[str] = []
    pred_rows: List[Dict[str, Any]] = []

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # compute loss too, so you can inspect test loss if needed
        with processor.as_target_processor():
            label_features = processor.tokenizer(
                batch["references"],
                padding=True,
                return_tensors="pt",
            )
        labels = label_features["input_ids"]
        labels = labels.masked_fill(label_features["attention_mask"].ne(1), -100).to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += float(outputs.loss.item())
        total_batches += 1

        pred_ids = torch.argmax(outputs.logits, dim=-1)
        preds = processor.batch_decode(pred_ids)
        refs = batch["references"]

        preds = [clean_text(p) for p in preds]
        refs = [clean_text(r) for r in refs]

        all_hyps.extend(preds)
        all_refs.extend(refs)

        for audio_path, ref, hyp in zip(batch["audio_paths"], refs, preds):
            pred_rows.append({
                "audio_path": audio_path,
                "reference": ref,
                "prediction": hyp,
            })

    metrics = compute_metrics(all_refs, all_hyps)
    metrics["avg_test_loss"] = total_loss / max(1, total_batches)
    metrics["num_examples"] = len(all_refs)

    return {
        "metrics": metrics,
        "predictions": pred_rows,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate fra-eng ASR model on test_pool.json")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model dir OR adapter dir")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Required only if model_path is a LoRA adapter dir")
    parser.add_argument("--test_manifest", type=str, required=True,
                        help="Path to test_pool.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write eval_results.json and predictions.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = choose_device(args.device)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "eval_test")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1] Loading model from: {args.model_path}")
    if is_adapter_dir(args.model_path):
        print("    Detected PEFT adapter directory")
        print(f"    Base model: {args.base_model}")
    else:
        print("    Detected merged/full model directory")

    processor, model = load_model_and_processor(
        model_path=args.model_path,
        base_model=args.base_model,
        device=device,
    )

    print(f"[2] Evaluating on: {args.test_manifest}")
    results = evaluate_model(
        model=model,
        processor=processor,
        manifest_path=args.test_manifest,
        batch_size=args.batch_size,
        device=device,
    )

    metrics_path = os.path.join(args.output_dir, "eval_results.json")
    preds_path = os.path.join(args.output_dir, "predictions.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, ensure_ascii=False, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(results["predictions"], f, ensure_ascii=False, indent=2)

    print("\n[3] Done")
    print(json.dumps(results["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()