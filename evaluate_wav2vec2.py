import os
import io
import json
import torch
import librosa
import numpy as np
import argparse
import soundfile as sf
from typing import List, Dict, Union
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from evaluate import load
from tqdm import tqdm
from huggingface_hub import snapshot_download

def load_audio_from_raw(val: dict, target_sr: int = 16000):
    """Decode audio from a raw (decode=False) dict with 'bytes' and/or 'path' keys.
    Returns (numpy_array, sample_rate) or (None, None) on failure.
    """
    raw_bytes = val.get("bytes") if isinstance(val, dict) else None
    path = val.get("path") if isinstance(val, dict) else None
    if raw_bytes:
        try:
            audio, sr = sf.read(io.BytesIO(raw_bytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # stereo -> mono
            if sr != target_sr:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            return audio.astype(np.float32), target_sr
        except Exception:
            try:
                audio, sr = librosa.load(io.BytesIO(raw_bytes), sr=target_sr)
                return audio, target_sr
            except Exception:
                pass
    if path:
        try:
            audio, sr = librosa.load(path, sr=target_sr)
            return audio, target_sr
        except Exception:
            pass
    return None, None

def clean_text(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()

def evaluate(model, processor, dataset, device, max_samples=None):
    wer_metric = load("wer")
    cer_metric = load("cer")
    
    predictions = []
    references = []
    
    samples = dataset
    if max_samples and not hasattr(dataset, "__iter__") and hasattr(dataset, "select"):
        samples = dataset.select(range(min(len(dataset), max_samples)))
    
    model.to(device)
    model.eval()
    
    for i, batch in enumerate(tqdm(samples, desc="Evaluating", total=max_samples)):
        if max_samples and i >= max_samples:
            break
        # Try multiple keys for audio data
        audio = None
        sampling_rate = 16000
        
        # Order: 'audio' (HF standard), 'context' (SEAME), 'input_values' (custom), 'correct_audio' (ICASSP 2024)
        for audio_key in ["audio", "context", "input_values", "speech", "correct_audio"]:
            if audio_key in batch:
                val = batch[audio_key]
                if isinstance(val, dict):
                    if "array" in val and val["array"] is not None:
                        # Standard decoded audio dict
                        audio = np.array(val["array"])
                        sampling_rate = val.get("sampling_rate", 16000)
                    elif "bytes" in val or "path" in val:
                        # Raw (decode=False) audio dict — decode manually
                        audio, sampling_rate = load_audio_from_raw(val)
                elif isinstance(val, (np.ndarray, list)):
                    audio = np.array(val)
                break
        
        if audio is None:
            print(f"Warning: No audio found in batch {i}. Keys available: {list(batch.keys())}")
            continue

        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        predictions.append(transcription)
        
        # Determine reference text key
        ref_text = ""
        for key in ["text", "sentence", "transcription", "answer", "correct_transcription"]:
            if key in batch:
                ref_text = batch[key]
                break
        
        if not ref_text:
            print(f"Warning: No transcription found in batch {i}")
        
        references.append(clean_text(ref_text))

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    return wer, cer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate per dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print(f"Loading model and processor from {args.model_path}...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)

    # 1) Evaluate on CS FLEURS (test subset or split from our training script logic)
    print("\n--- Evaluating on CS FLEURS (spa-eng) ---")
    # Using the same snapshot logic to get metadata
    snapshot_path = snapshot_download(
        "byan/cs-fleurs",
        repo_type="dataset",
        # allow_patterns=["xtts/train/metadata.jsonl", "xtts/train/audio/cs_cmn_eng_n1_resample/*"]
        allow_patterns=["xtts/train/metadata.jsonl", "xtts/train/audio/cs_spa_eng_n1_resample/*"]

    )
    metadata_path = os.path.join(snapshot_path, "xtts/train/metadata.jsonl")
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("language") == "spa-eng":
                entries.append(obj)
    
    # Simple fixed split for evaluation if not specified differently by user
    # Here we just take the last 100 samples as 'eval'
    eval_entries = entries[-max(100, args.max_samples):]
    
    # Custom loader for CS-FLEURS subset
    base_audio_path = os.path.join(snapshot_path, "xtts/train")
    cs_fleurs_data = []
    for item in eval_entries[:args.max_samples]:
        rel_path = item["file_name"].replace("//", "/")
        audio_path = os.path.join(base_audio_path, rel_path)
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            cs_fleurs_data.append({"input_values": audio, "text": item["text"]})
        except Exception as e:
            continue

    if cs_fleurs_data:
        wer_fleurs,cer_fleurs = evaluate(model, processor, cs_fleurs_data, args.device)
        print(f"CS FLEURS WER: {wer_fleurs:.4f}")
        print(f"CS FLEURS CER: {cer_fleurs:.4f}")
    else:
        print("No CS FLEURS samples found for evaluation.")

    # 2) Evaluate on SEAME
    # print("\n--- Evaluating on SEAME (AudioLLMs/seame_dev_sge) ---")
    # try:
    #     from datasets import Audio
    #     seame_dataset = load_dataset("AudioLLMs/seame_dev_sge", split="test")
    #     # Ensure 'context' is treated as audio if it exists and is not already cast
    #     if "context" in seame_dataset.column_names:
    #         seame_dataset = seame_dataset.cast_column("context", Audio(sampling_rate=16000, decode=False))
    #     # Get the full duration in hours
    #     # total_duration = sum([float(e["duration"]) for e in seame_dataset]) / 3600
    #     # print(f"Total duration: {total_duration} hours")    
    #     wer_seame,cer_seame = evaluate(model, processor, seame_dataset, args.device, max_samples=args.max_samples)
    #     print(f"SEAME WER: {wer_seame:.4f}")
    #     print(f"SEAME CER: {cer_seame:.4f}")
    # except Exception as e:
    #     print(f"Error loading SEAME dataset: {e}")

    # 3) Evaluate on Spanish-English track of the benchmark from ICASSP 2024
    print("\n--- Evaluating on Spanish-English track of the benchmark from ICASSP 2024Corpus (ky552/cszs_es_en) ---")
    try:
        # Use streaming=True with decode=False on audio columns to avoid torchcodec/FFmpeg dependency.
        # Audio bytes are then decoded manually in the evaluate() loop via load_audio_from_raw().
        args.max_samples = len(load_dataset("ky552/cszs_es_en", split="test"))  
        spa_eng_test = (
            load_dataset("ky552/cszs_es_en", split="test", streaming=True)
            .cast_column("correct_audio", Audio(decode=False))
            .cast_column("wrong_audio", Audio(decode=False))
            .take(args.max_samples)
        )
        wer_spa_eng_test, cer_spa_eng_test = evaluate(model, processor, spa_eng_test, args.device, max_samples=args.max_samples)
        print(f"Spanish-English track of the benchmark from ICASSP 2024 Corpus WER: {wer_spa_eng_test:.4f}")
        print(f"Spanish-English track of the benchmark from ICASSP 2024 Corpus CER: {cer_spa_eng_test:.4f}")
    except Exception as e:
        print(f"Error loading Spanish-English track of the benchmark from ICASSP 2024Corpus: {e}")

if __name__ == "__main__":
    main()

#python evaluate_wav2vec2.py --model_path finetuned_from_1h_meta/ --device cuda