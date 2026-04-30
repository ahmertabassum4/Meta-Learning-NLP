import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import torch
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import re


def clean_text(text: str) -> str:
    text = text.replace("**", "").replace("__", "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_model_and_processor(model_path: str, device: str):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"
    model.generation_config.language = "bn"
    model.generation_config.max_length = None

    model.to(device)
    model.eval()
    return model, processor


def read_manifest_csv(manifest_path: str) -> List[Dict]:
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def resolve_audio_path(split_dir: str, row: Dict) -> str:
    for key in ["audio", "audio_path", "file_name", "path", "wav", "filename"]:
        if key in row and row[key]:
            rel = row[key].strip()
            return rel if os.path.isabs(rel) else os.path.join(split_dir, rel)
    raise KeyError(
        "Could not find audio path column in manifest.csv. "
        "Expected one of: audio, audio_path, file_name, path, wav, filename"
    )


def resolve_text(row: Dict) -> str:
    for key in ["text", "transcript", "sentence", "normalized_text"]:
        if key in row and row[key]:
            return clean_text(row[key])
    raise KeyError(
        "Could not find transcript column in manifest.csv. "
        "Expected one of: text, transcript, sentence, normalized_text"
    )


def transcribe_audio(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio: np.ndarray,
    device: str,
    target_sr: int = 16000,
    max_new_tokens: int = 225,
    num_beams: int = 1,
) -> str:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    inputs = processor(
        audio,
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return clean_text(pred)


def compute_wer_cer(predictions: List[str], references: List[str]) -> Tuple[float, float]:
    wer_metric = load("wer")
    cer_metric = load("cer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    return wer, cer


def evaluate_repo_split(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    split_dir: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 225,
    num_beams: int = 1,
) -> Dict:
    manifest_path = os.path.join(split_dir, "manifest.csv")
    entries = read_manifest_csv(manifest_path)

    if max_samples is not None:
        entries = entries[:max_samples]

    predictions = []
    references = []
    skipped = 0
    examples = []

    for item in tqdm(entries, desc=f"Repo split eval: {os.path.basename(split_dir)}"):
        try:
            audio_path = resolve_audio_path(split_dir, item)
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)

            pred = transcribe_audio(
                model=model,
                processor=processor,
                audio=audio,
                device=device,
                target_sr=16000,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            ref = resolve_text(item)

            predictions.append(pred)
            references.append(ref)

            examples.append({
                "id": item.get("id", ""),
                "audio_path": audio_path,
                "reference": ref,
                "prediction": pred,
            })

        except Exception as e:
            skipped += 1
            print(f"[RepoEval] Skipping sample: {e}")

    if not predictions:
        raise RuntimeError("No repo test samples were successfully evaluated.")

    wer, cer = compute_wer_cer(predictions, references)

    print("\nSample repo-test predictions:")
    for i, ex in enumerate(examples[:10], 1):
        print(f"\n--- Example {i} ---")
        print("ID  :", ex["id"])
        print("REF :", ex["reference"])
        print("PRED:", ex["prediction"])

    return {
        "dataset": f"repo_{os.path.basename(split_dir)}",
        "num_scored": len(predictions),
        "num_skipped": skipped,
        "wer": wer,
        "cer": cer,
        "examples": examples[:20],
    }


def read_kaldi_wav_scp(path: str) -> Dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                continue
            rec_id, wav_path = parts
            mapping[rec_id] = wav_path
    return mapping


def read_kaldi_segments(path: str) -> List[Tuple[str, str, float, float]]:
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            utt_id, rec_id, start, end = parts
            segments.append((utt_id, rec_id, float(start), float(end)))
    return segments


def read_kaldi_text(path: str) -> Dict[str, str]:
    texts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            texts[utt_id] = text
    return texts


def resolve_audio_path_kaldi(wav_root: str, wav_rel_or_name: str) -> str:
    if os.path.isabs(wav_rel_or_name):
        return wav_rel_or_name
    return os.path.join(wav_root, wav_rel_or_name)


def evaluate_openslr104_kaldi(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    wav_root: str,
    wav_scp_path: str,
    segments_path: str,
    text_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 225,
    num_beams: int = 1,
) -> Dict:
    wav_map = read_kaldi_wav_scp(wav_scp_path)
    segments = read_kaldi_segments(segments_path)
    text_map = read_kaldi_text(text_path)

    if max_samples is not None:
        segments = segments[:max_samples]

    predictions = []
    references = []
    skipped = 0
    examples = []

    for utt_id, rec_id, start, end in tqdm(segments, desc="OpenSLR104 real-world test"):
        if utt_id not in text_map:
            skipped += 1
            print(f"[OpenSLR104] Missing transcript for utterance: {utt_id}")
            continue
        if rec_id not in wav_map:
            skipped += 1
            print(f"[OpenSLR104] Missing wav.scp entry for recording: {rec_id}")
            continue

        audio_path = resolve_audio_path_kaldi(wav_root, wav_map[rec_id])

        try:
            duration = max(0.0, end - start)
            audio, _ = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
                offset=start,
                duration=duration,
            )
            if audio is None or len(audio) == 0:
                raise RuntimeError("empty segment")

            pred = transcribe_audio(
                model=model,
                processor=processor,
                audio=audio,
                device=device,
                target_sr=16000,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            ref = clean_text(text_map[utt_id])

            predictions.append(pred)
            references.append(ref)

            examples.append({
                "utt_id": utt_id,
                "audio_path": audio_path,
                "reference": ref,
                "prediction": pred,
            })

        except Exception as e:
            skipped += 1
            print(f"[OpenSLR104] Skipping {utt_id} from {audio_path}: {e}")

    if not predictions:
        raise RuntimeError("No OpenSLR104 samples were successfully evaluated.")

    wer, cer = compute_wer_cer(predictions, references)

    print("\nSample OpenSLR104 predictions:")
    for i, ex in enumerate(examples[:10], 1):
        print(f"\n--- Example {i} ---")
        print("UTT :", ex["utt_id"])
        print("REF :", ex["reference"])
        print("PRED:", ex["prediction"])

    return {
        "dataset": "openslr104_ben_eng_real_test",
        "num_scored": len(predictions),
        "num_skipped": skipped,
        "wer": wer,
        "cer": cer,
        "examples": examples[:30],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--repo_test_dir",
        type=str,
        required=True,
        help="Path like data/cs_fleurs_ben_eng_300_49/test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--max_samples_repo", type=int, default=None)
    parser.add_argument("--max_samples_real", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--real_wav_root", type=str, required=True)
    parser.add_argument("--real_wav_scp", type=str, required=True)
    parser.add_argument("--real_segments", type=str, required=True)
    parser.add_argument("--real_text", type=str, required=True)

    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model, processor = load_model_and_processor(args.model_path, args.device)

    print(f"\n--- Evaluating repo test split ({args.repo_test_dir}) ---")
    repo_result = evaluate_repo_split(
        model=model,
        processor=processor,
        device=args.device,
        split_dir=args.repo_test_dir,
        max_samples=args.max_samples_repo,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(json.dumps(repo_result, ensure_ascii=False, indent=2))

    print("\n--- Evaluating OpenSLR104 real-world Bengali-English test ---")
    real_result = evaluate_openslr104_kaldi(
        model=model,
        processor=processor,
        device=args.device,
        wav_root=args.real_wav_root,
        wav_scp_path=args.real_wav_scp,
        segments_path=args.real_segments,
        text_path=args.real_text,
        max_samples=args.max_samples_real,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(json.dumps(real_result, ensure_ascii=False, indent=2))

    results = {
        "model_path": args.model_path,
        "repo_test": repo_result,
        "openslr104_real": real_result,
    }

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()