import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import torch
from evaluate import load
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm


def clean_text(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def load_model_and_processor(model_path: str, device: str):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.to(device)
    model.eval()
    return model, processor


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


def load_csfleurs_metadata(snapshot_path: str, subset: str, lang_pair: str) -> List[Dict]:
    metadata_path = os.path.join(snapshot_path, subset, "metadata.jsonl")
    entries = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("language") == lang_pair:
                entries.append(obj)

    return entries


def resolve_csfleurs_subset(subset_name: str) -> str:
    subset_map = {
        "xtts_test1": "xtts/test1",
    }
    if subset_name not in subset_map:
        raise ValueError(f"Unsupported CS-FLEURS subset: {subset_name}")
    return subset_map[subset_name]


def evaluate_csfleurs(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    lang_pair: str = "hin-eng",
    subset_name: str = "xtts_test1",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 225,
    num_beams: int = 1,
) -> Dict:
    subset_rel = resolve_csfleurs_subset(subset_name)

    snapshot_path = snapshot_download(
        repo_id="byan/cs-fleurs",
        repo_type="dataset",
        allow_patterns=[f"{subset_rel}/metadata.jsonl"],
    )

    entries = load_csfleurs_metadata(snapshot_path, subset_rel, lang_pair)
    if not entries:
        raise ValueError(
            f"No CS-FLEURS entries found for lang_pair='{lang_pair}' in subset='{subset_name}'."
        )

    if max_samples is not None:
        entries = entries[:max_samples]

    audio_patterns = [
        f"{subset_rel}/{item['file_name'].replace('//', '/')}"
        for item in entries
    ]

    snapshot_path = snapshot_download(
        repo_id="byan/cs-fleurs",
        repo_type="dataset",
        allow_patterns=[f"{subset_rel}/metadata.jsonl"] + audio_patterns,
    )

    predictions = []
    references = []
    skipped = 0

    for item in tqdm(entries, desc=f"CS-FLEURS {subset_name}"):
        rel_path = item["file_name"].replace("//", "/")
        audio_path = os.path.join(snapshot_path, subset_rel, rel_path)

        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            pred = transcribe_audio(
                model, processor, audio, device,
                target_sr=16000,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            ref = clean_text(item["text"])
            predictions.append(pred)
            references.append(ref)
        except Exception as e:
            skipped += 1
            print(f"[CS-FLEURS] Skipping {audio_path}: {e}")

    if not predictions:
        raise RuntimeError("No CS-FLEURS samples were successfully evaluated.")

    wer, cer = compute_wer_cer(predictions, references)
    return {
        "dataset": f"csfleurs_{subset_name}",
        "lang_pair": lang_pair,
        "num_scored": len(predictions),
        "num_skipped": skipped,
        "wer": wer,
        "cer": cer,
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


def resolve_audio_path(wav_root: str, wav_rel_or_name: str) -> str:
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

    for utt_id, rec_id, start, end in tqdm(segments, desc="OpenSLR104 real-world test"):
        if utt_id not in text_map:
            skipped += 1
            print(f"[OpenSLR104] Missing transcript for utterance: {utt_id}")
            continue
        if rec_id not in wav_map:
            skipped += 1
            print(f"[OpenSLR104] Missing wav.scp entry for recording: {rec_id}")
            continue

        audio_path = resolve_audio_path(wav_root, wav_map[rec_id])

        try:
            duration = max(0.0, end - start)
            audio, _ = librosa.load(audio_path, sr=16000, mono=True, offset=start, duration=duration)
            if audio is None or len(audio) == 0:
                raise RuntimeError("empty segment")

            pred = transcribe_audio(
                model, processor, audio, device,
                target_sr=16000,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            ref = clean_text(text_map[utt_id])
            predictions.append(pred)
            references.append(ref)
        except Exception as e:
            skipped += 1
            print(f"[OpenSLR104] Skipping {utt_id} from {audio_path}: {e}")

    if not predictions:
        raise RuntimeError("No OpenSLR104 samples were successfully evaluated.")

    wer, cer = compute_wer_cer(predictions, references)
    return {
        "dataset": "openslr104_hin_eng_real_test",
        "num_scored": len(predictions),
        "num_skipped": skipped,
        "wer": wer,
        "cer": cer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lang_pair", type=str, default="hin-eng")
    parser.add_argument("--csfleurs_subset", type=str, default="xtts_test1", choices=["xtts_test1"])
    parser.add_argument("--max_samples_csfleurs", type=int, default=None)
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

    print(f"\n--- Evaluating CS-FLEURS ({args.lang_pair}, {args.csfleurs_subset}) ---")
    csfleurs_result = evaluate_csfleurs(
        model=model,
        processor=processor,
        device=args.device,
        lang_pair=args.lang_pair,
        subset_name=args.csfleurs_subset,
        max_samples=args.max_samples_csfleurs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(json.dumps(csfleurs_result, ensure_ascii=False, indent=2))

    print("\n--- Evaluating OpenSLR104 real-world Hindi-English test ---")
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
        "csfleurs": csfleurs_result,
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