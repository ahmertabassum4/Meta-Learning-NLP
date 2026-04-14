import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from evaluate import load
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm


def clean_text(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def load_model_and_processor(model_path: str, device: str):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def transcribe_audio(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    audio: np.ndarray,
    device: str,
    target_sr: int = 16000,
) -> str:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        pred_ids = torch.argmax(outputs.logits, dim=-1)

    pred = processor.batch_decode(pred_ids)[0]
    return clean_text(pred)


def compute_wer_cer(predictions: List[str], references: List[str]) -> Tuple[float, float]:
    wer_metric = load("wer")
    cer_metric = load("cer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    return wer, cer


def infer_csfleurs_audio_dir(lang_pair: str) -> str:
    return f"cs_{lang_pair.replace('-', '_')}_n1_resample"


def load_csfleurs_metadata(
    snapshot_path: str,
    subset: str,
    lang_pair: str,
) -> List[Dict]:
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
        # "xtts_test2": "xtts/test2",
        # "read_test": "read/test",
        # "mms_test": "mms/test",
    }
    if subset_name not in subset_map:
        raise ValueError(f"Unsupported CS-FLEURS subset: {subset_name}")
    return subset_map[subset_name]

def evaluate_csfleurs(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: str,
    lang_pair: str = "hin-eng",
    subset_name: str = "xtts_test1",
    max_samples: Optional[int] = None,
) -> Dict:
    subset_rel = resolve_csfleurs_subset(subset_name)

    # Step 1: download only metadata first
    snapshot_path = snapshot_download(
        repo_id="byan/cs-fleurs",
        repo_type="dataset",
        allow_patterns=[f"{subset_rel}/metadata.jsonl"],
    )

    # Step 2: filter metadata for the target language pair
    entries = load_csfleurs_metadata(snapshot_path, subset_rel, lang_pair)
    if not entries:
        raise ValueError(
            f"No CS-FLEURS entries found for lang_pair='{lang_pair}' in subset='{subset_name}'."
        )

    if max_samples is not None:
        entries = entries[:max_samples]

    # Step 3: build exact audio file paths from filtered metadata
    audio_patterns = [
        f"{subset_rel}/{item['file_name'].replace('//', '/')}"
        for item in entries
    ]

    # Step 4: download only the matching audio files
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
            audio, sr = librosa.load(audio_path, sr=16000)
            pred = transcribe_audio(model, processor, audio, device, target_sr=16000)
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
# def evaluate_csfleurs(
#     model: Wav2Vec2ForCTC,
#     processor: Wav2Vec2Processor,
#     device: str,
#     lang_pair: str = "hin-eng",
#     subset_name: str = "xtts_test1",
#     max_samples: Optional[int] = None,
# ) -> Dict:
#     subset_rel = resolve_csfleurs_subset(subset_name)

#     # Choose an audio folder layout that matches the chosen subset.
#     # For XTTS subsets, audio is nested by language-pair directory.
#     # For read/test, the dataset is organized by language directories and is not code-switched like XTTS.
#     xtts_audio_dir = infer_csfleurs_audio_dir(lang_pair)

#     allow_patterns = [f"{subset_rel}/metadata.jsonl"]

#     if subset_name.startswith("xtts_"):
#         allow_patterns.extend([
#             f"{subset_rel}/audio/*",
#             f"{subset_rel}/audio/*/*",
#         ])
#     # elif subset_name == "read_test":
#     #     allow_patterns.extend([
#     #         f"{subset_rel}/audio/*",
#     #         f"{subset_rel}/audio/*/*",
#     #     ])
#     # elif subset_name == "mms_test":
#     #     allow_patterns.extend([
#     #         f"{subset_rel}/audio/*",
#     #         f"{subset_rel}/audio/*/*",
#     #     ])

#     snapshot_path = snapshot_download(
#         repo_id="byan/cs-fleurs",
#         repo_type="dataset",
#         allow_patterns=allow_patterns,
#     )

#     entries = load_csfleurs_metadata(snapshot_path, subset_rel, lang_pair)
#     if not entries:
#         raise ValueError(
#             f"No CS-FLEURS entries found for lang_pair='{lang_pair}' in subset='{subset_name}'."
#         )

#     if max_samples is not None:
#         entries = entries[:max_samples]

#     predictions = []
#     references = []
#     skipped = 0

#     for item in tqdm(entries, desc=f"CS-FLEURS {subset_name}"):
#         rel_path = item["file_name"].replace("//", "/")

#         if subset_name.startswith("xtts_"):
#             audio_path = os.path.join(snapshot_path, subset_rel, rel_path)
#         else:
#             audio_path = os.path.join(snapshot_path, subset_rel, rel_path)

#         try:
#             audio, sr = librosa.load(audio_path, sr=16000)
#             pred = transcribe_audio(model, processor, audio, device, target_sr=16000)
#             ref = clean_text(item["text"])
#             predictions.append(pred)
#             references.append(ref)
#         except Exception as e:
#             skipped += 1
#             print(f"[CS-FLEURS] Skipping {audio_path}: {e}")

#     if not predictions:
#         raise RuntimeError("No CS-FLEURS samples were successfully evaluated.")

#     wer, cer = compute_wer_cer(predictions, references)
#     return {
#         "dataset": f"csfleurs_{subset_name}",
#         "lang_pair": lang_pair,
#         "num_scored": len(predictions),
#         "num_skipped": skipped,
#         "wer": wer,
#         "cer": cer,
#     }


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
    # wav.scp may contain bare filenames like "abc.wav" or longer paths.
    if os.path.isabs(wav_rel_or_name):
        return wav_rel_or_name
    return os.path.join(wav_root, wav_rel_or_name)


def evaluate_openslr104_kaldi(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: str,
    wav_root: str,
    wav_scp_path: str,
    segments_path: str,
    text_path: str,
    max_samples: Optional[int] = None,
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
            audio, sr = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
            if audio is None or len(audio) == 0:
                raise RuntimeError("empty segment")
            pred = transcribe_audio(model, processor, audio, device, target_sr=16000)
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
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lang_pair", type=str, default="hin-eng")
    parser.add_argument(
        "--csfleurs_subset",
        type=str,
        default="xtts_test1",
        choices=["xtts_test1"],
        help="Which CS-FLEURS test subset to use",
    )
    parser.add_argument("--max_samples_csfleurs", type=int, default=None)
    parser.add_argument("--max_samples_real", type=int, default=None)

    # OpenSLR104 real-world test paths
    parser.add_argument("--real_wav_root", type=str, required=True, help="Directory containing the extracted test WAV files")
    parser.add_argument("--real_wav_scp", type=str, required=True, help="Path to wav.scp")
    parser.add_argument("--real_segments", type=str, required=True, help="Path to segments")
    parser.add_argument("--real_text", type=str, required=True, help="Path to text")

    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save results JSON")
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


    """
    python ahmer/evaluate_hin-eng.py   --model_path /workspace/projects/asr_project_src/runs/w2v2_csfleurs_hin_eng   --device cuda   --lang_pair hin-eng   --csfleurs_subset xtts_test1   --real_wav_root /workspace/projects/asr_project_src/test   --real_wav_scp /workspace/projects/asr_project_src/test/transcripts/wav.scp   --real_segments /workspace/projects/asr_project_src/test/transcripts/segments   --real_text /workspace/projects/asr_project_src/test/transcripts/text   --save_json /workspace/projects/asr_project_src/runs/w2v2_csfleurs_hin_eng/eval_results.json
    """