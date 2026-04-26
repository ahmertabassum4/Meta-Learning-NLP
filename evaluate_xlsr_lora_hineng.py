#!/usr/bin/env python3
"""
Evaluate a Hindi-English code-switched XLS-R LoRA ASR model.

Compatible with the training script that:
  - builds a character vocab from train transcripts
  - uses Wav2Vec2CTCTokenizer with word_delimiter_token="|"
  - stores processor files inside the run directory
  - optionally saves a merged model under <run_dir>/merged

Features:
  1. Parses Kaldi-style test data
  2. Loads either:
       - a LoRA adapter run directory, or
       - a merged standalone model directory
  3. Runs greedy CTC decoding in batches
  4. Computes WER and CER using evaluate
  5. Reports blank-dominance diagnostics
  6. Reports empty-hypothesis rate
  7. Reports character OOV coverage against the trained tokenizer vocab
  8. Saves per-utterance predictions to JSONL

Expected test layout:

    <test_root>/
        transcripts/
            text
            wav.scp
            segments
            utt2spk   # optional / unused
        audio/        # optional, if wav.scp entries need resolution from here
        *.wav         # also supported if wav.scp basenames point to dataset root

Examples:
  Adapter evaluation:
    python evaluate_xlsr_lora_hineng.py \
      --test_root /path/to/test-SLR-hin \
      --adapter_dir runs/xlsr_lora_hineng_4hr \
      --base_model facebook/wav2vec2-xls-r-300m \
      --batch_size 8 \
      --out_jsonl runs/xlsr_lora_hineng_4hr/test_predictions.jsonl

  Merged model evaluation:
    python evaluate_xlsr_lora_hineng.py \
      --test_root /path/to/test-SLR-hin \
      --merged_dir runs/xlsr_lora_hineng_4hr/merged \
      --batch_size 8

Notes:
  - Use either --adapter_dir or --merged_dir.
  - For adapter_dir, point to the run directory that contains:
        adapter_config.json
        adapter_model.safetensors
        tokenizer_config.json
        vocab.json
    exactly like in your screenshot.
"""

import os
import json
import argparse
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import librosa

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import PeftModel
import evaluate


# ============================================================================ #
#  Kaldi parsing                                                                #
# ============================================================================ #

def read_kaldi_two_col(path: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            out[parts[0]] = parts[1] if len(parts) == 2 else ""
    return out


def read_segments(path: str) -> Dict[str, Dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            utt_id, rec_id = parts[0], parts[1]
            start, end = float(parts[2]), float(parts[3])
            if end <= start:
                continue
            out[utt_id] = {"rec_id": rec_id, "start": start, "end": end}
    return out


def _index_audio_dir(root_dir: str) -> Dict[str, str]:
    index = {}
    if not os.path.isdir(root_dir):
        return index
    for root, _, files in os.walk(root_dir):
        for fn in files:
            full = os.path.join(root, fn)
            index[fn] = full
            stem, _ = os.path.splitext(fn)
            index.setdefault(stem, full)
    return index


def resolve_wav_path(
    scp_value: str,
    test_root: str,
    audio_index: Dict[str, str],
) -> Optional[str]:
    val = scp_value.strip()
    if not val:
        return None

    # Kaldi pipe — try to salvage an audio file token
    if val.endswith("|"):
        for tok in val.split():
            if tok.endswith(".wav") or tok.endswith(".flac"):
                base = os.path.basename(tok)
                hit = audio_index.get(base) or audio_index.get(os.path.splitext(base)[0])
                if hit:
                    return hit
        return None

    # Absolute path
    if os.path.isabs(val) and os.path.exists(val):
        return val

    # Relative to test root
    cand = os.path.join(test_root, val)
    if os.path.exists(cand):
        return cand

    # Basename fallback
    base = os.path.basename(val)
    return audio_index.get(base) or audio_index.get(os.path.splitext(base)[0])


def build_entries(test_root: str, transcripts_dir: str) -> List[Dict]:
    tdir = os.path.join(test_root, transcripts_dir)

    text = read_kaldi_two_col(os.path.join(tdir, "text"))
    wav_scp = read_kaldi_two_col(os.path.join(tdir, "wav.scp"))

    segments_path = os.path.join(tdir, "segments")
    if not os.path.exists(segments_path):
        raise RuntimeError(f"Missing segments file: {segments_path}")
    segments = read_segments(segments_path)

    # Index both test_root and test_root/audio to be flexible
    audio_index = {}
    audio_index.update(_index_audio_dir(test_root))
    audio_index.update(_index_audio_dir(os.path.join(test_root, "audio")))
    print(f"[data] Indexed {len(audio_index)} audio basenames under test root")

    entries = []
    missing_seg = 0
    missing_rec_in_scp = 0
    unresolved_path = 0
    bad_examples = []

    for utt_id, transcription in text.items():
        seg = segments.get(utt_id)
        if seg is None:
            missing_seg += 1
            continue

        rec_id = seg["rec_id"]
        start, end = seg["start"], seg["end"]

        wav_val = wav_scp.get(rec_id)
        if wav_val is None:
            missing_rec_in_scp += 1
            continue

        wav_path = resolve_wav_path(wav_val, test_root, audio_index)
        if wav_path is None:
            unresolved_path += 1
            if len(bad_examples) < 5:
                bad_examples.append(
                    f"  utt={utt_id} rec={rec_id} scp_value={wav_val!r}"
                )
            continue

        entries.append(
            {
                "utt_id": utt_id,
                "rec_id": rec_id,
                "text": transcription,
                "wav_path": wav_path,
                "start": start,
                "end": end,
            }
        )

    print(
        f"[data] Resolved {len(entries)}/{len(text)} utterances "
        f"(dropped: missing_seg={missing_seg}, rec_not_in_scp={missing_rec_in_scp}, "
        f"file_not_found={unresolved_path})"
    )

    if bad_examples:
        print("[data] Examples of unresolved paths:")
        for ex in bad_examples:
            print(ex)

    if not entries:
        raise RuntimeError("Zero usable utterances found in test set.")
    return entries


# ============================================================================ #
#  Text normalization                                                            #
# ============================================================================ #

def clean_text(text: str) -> str:
    # Match training behavior exactly
    return text.replace("**", "").replace("__", "").strip()


def normalize_text_for_labels(text: str) -> str:
    # Match training tokenizer label construction exactly
    return clean_text(text).replace(" ", "|")


def normalize_text_for_metrics(text: str) -> str:
    # Metric text should stay human-readable with spaces
    return clean_text(text)


# ============================================================================ #
#  Audio                                                                        #
# ============================================================================ #

def load_clip(wav_path: str, start: float, end: float, target_sr: int = 16000) -> np.ndarray:
    duration = max(1e-3, end - start)
    audio, _ = librosa.load(
        wav_path,
        sr=target_sr,
        offset=start,
        duration=duration,
    )
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return np.asarray(audio, dtype=np.float32)


# ============================================================================ #
#  Model loading                                                                 #
# ============================================================================ #

def load_model_and_processor(args, device: str):
    if args.merged_dir and args.adapter_dir:
        raise ValueError("Use only one of --merged_dir or --adapter_dir")

    if not args.merged_dir and not args.adapter_dir:
        raise ValueError("Pass one of --merged_dir or --adapter_dir")

    if args.merged_dir:
        print(f"[load] Loading merged model from: {args.merged_dir}")
        processor = Wav2Vec2Processor.from_pretrained(args.merged_dir)
        model = Wav2Vec2ForCTC.from_pretrained(args.merged_dir)
    else:
        print(f"[load] Loading adapter from: {args.adapter_dir}")
        print(f"[load] Base model: {args.base_model}")

        processor = Wav2Vec2Processor.from_pretrained(args.adapter_dir)

        base = Wav2Vec2ForCTC.from_pretrained(
            args.base_model,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
            ctc_zero_infinity=True,
        )

        model = PeftModel.from_pretrained(base, args.adapter_dir)

        if args.merge_for_inference:
            try:
                model = model.merge_and_unload()
                print("[load] Adapter merged into base for inference")
            except Exception as exc:
                print(f"[load] merge_and_unload failed, continuing without merge: {exc}")

    model.to(device)
    model.eval()
    return model, processor


# ============================================================================ #
#  Diagnostics                                                                   #
# ============================================================================ #

def vocab_coverage_report(texts: List[str], processor) -> Dict:
    vocab = set(processor.tokenizer.get_vocab().keys())
    vocab.discard("[PAD]")
    vocab.discard("[UNK]")

    counted_chars = {}
    oov_chars = {}

    for t in texts:
        norm = normalize_text_for_labels(t)
        for ch in norm:
            if ch in ("[", "]"):
                # impossible here for normal text, defensive only
                continue
            counted_chars[ch] = counted_chars.get(ch, 0) + 1
            if ch not in vocab:
                oov_chars[ch] = oov_chars.get(ch, 0) + 1

    total = sum(counted_chars.values()) or 1
    oov_total = sum(oov_chars.values())
    return {
        "total_chars": total,
        "oov_chars": oov_total,
        "oov_pct": 100.0 * oov_total / total,
        "oov_breakdown": dict(sorted(oov_chars.items(), key=lambda kv: -kv[1])),
    }


def has_only_in_vocab_chars(text: str, processor) -> bool:
    vocab = set(processor.tokenizer.get_vocab().keys())
    norm = normalize_text_for_labels(text)
    return all(ch in vocab for ch in norm)


def blank_dominance_stats(pred_ids: torch.Tensor, blank_id: int) -> List[float]:
    stats = []
    arr = pred_ids.detach().cpu().numpy()
    for seq in arr:
        frac = float((seq == blank_id).sum()) / max(1, len(seq))
        stats.append(frac)
    return stats


# ============================================================================ #
#  Inference                                                                     #
# ============================================================================ #

@torch.no_grad()
def transcribe_batch(model, processor, wavs: List[np.ndarray], device: str):
    inputs = processor(
        wavs,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    logits = model(
        input_values=input_values,
        attention_mask=attention_mask,
    ).logits

    pred_ids = torch.argmax(logits, dim=-1)

    pred_texts = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )

    return pred_texts, pred_ids


# ============================================================================ #
#  Main                                                                          #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hindi-English XLS-R LoRA / merged ASR model"
    )

    parser.add_argument("--test_root", type=str, required=True,
                        help="Path to test dataset root")
    parser.add_argument("--transcripts_dir", type=str, default="transcripts",
                        help="Relative path under test_root containing text/wav.scp/segments")

    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Run dir containing adapter_config.json, adapter_model.safetensors, vocab, tokenizer files")
    parser.add_argument("--merged_dir", type=str, default=None,
                        help="Merged model dir, e.g. runs/xlsr_lora_hineng_4hr/merged")
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--merge_for_inference", action="store_true",
                        help="When using adapter_dir, merge into base model for inference")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu / mps; default auto")
    parser.add_argument("--min_seg_sec", type=float, default=0.0)
    parser.add_argument("--max_seg_sec", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, evaluate only first N utterances")
    parser.add_argument("--out_jsonl", type=str, default=None,
                        help="Optional path to save per-utterance predictions")

    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[info] Using device: {device}")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # ------------------------------------------------------------------ #
    # Load test data                                                     #
    # ------------------------------------------------------------------ #
    print(f"[data] Reading test data from {os.path.join(args.test_root, args.transcripts_dir)}")
    entries = build_entries(args.test_root, args.transcripts_dir)
    print(f"[data] Loaded {len(entries)} raw utterances")

    before = len(entries)
    entries = [
        e for e in entries
        if args.min_seg_sec <= (e["end"] - e["start"]) <= args.max_seg_sec
    ]
    print(f"[data] Duration filter [{args.min_seg_sec}s, {args.max_seg_sec}s]: {before} -> {len(entries)}")

    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"[data] --limit applied: {len(entries)} utterances")

    if not entries:
        raise RuntimeError("No utterances left after filtering.")

    # ------------------------------------------------------------------ #
    # Load model                                                         #
    # ------------------------------------------------------------------ #
    model, processor = load_model_and_processor(args, device)

    # blank token for Wav2Vec2 CTC is pad_token_id
    blank_id = processor.tokenizer.pad_token_id
    print(f"[info] Tokenizer size: {len(processor.tokenizer)}")
    print(f"[info] pad_token_id / CTC blank id: {blank_id}")

    # ------------------------------------------------------------------ #
    # Vocab coverage                                                     #
    # ------------------------------------------------------------------ #
    refs_for_coverage = [normalize_text_for_metrics(e["text"]) for e in entries]
    cov = vocab_coverage_report(refs_for_coverage, processor)

    print(
        f"[vocab] total chars in refs: {cov['total_chars']} | "
        f"OOV chars: {cov['oov_chars']} ({cov['oov_pct']:.2f}%)"
    )
    if cov["oov_breakdown"]:
        top_items = list(cov["oov_breakdown"].items())[:20]
        print(f"[vocab] top OOV chars: {top_items}")

    # ------------------------------------------------------------------ #
    # Batched inference                                                  #
    # ------------------------------------------------------------------ #
    preds = []
    refs = []
    per_utt = []

    empty_count = 0
    load_failures = 0

    for start_idx in range(0, len(entries), args.batch_size):
        batch_entries = entries[start_idx:start_idx + args.batch_size]
        wavs = []
        ok_mask = []

        for e in batch_entries:
            try:
                wav = load_clip(e["wav_path"], e["start"], e["end"], target_sr=16000)
                if wav.size == 0:
                    raise RuntimeError("Loaded empty waveform")
                wavs.append(wav)
                ok_mask.append(True)
            except Exception as exc:
                print(f"[warn] Failed to load utt={e['utt_id']} path={e['wav_path']}: {exc}")
                wavs.append(np.zeros(16000, dtype=np.float32))
                ok_mask.append(False)
                load_failures += 1

        try:
            batch_preds, batch_pred_ids = transcribe_batch(model, processor, wavs, device)
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[warn] OOM at batch starting {start_idx}; falling back to batch_size=1")
            batch_preds = []
            pred_id_list = []
            for wav in wavs:
                one_pred, one_pred_ids = transcribe_batch(model, processor, [wav], device)
                batch_preds.extend(one_pred)
                pred_id_list.append(one_pred_ids[0].unsqueeze(0))
            batch_pred_ids = torch.cat(pred_id_list, dim=0)

        batch_blank_fracs = blank_dominance_stats(batch_pred_ids, blank_id=blank_id)

        for e, hyp, blank_frac, loaded_ok in zip(batch_entries, batch_preds, batch_blank_fracs, ok_mask):
            ref = normalize_text_for_metrics(e["text"])
            hyp = clean_text(hyp)

            if hyp == "":
                empty_count += 1

            preds.append(hyp)
            refs.append(ref)

            per_utt.append({
                "utt_id": e["utt_id"],
                "rec_id": e["rec_id"],
                "wav_path": e["wav_path"],
                "start": round(float(e["start"]), 3),
                "end": round(float(e["end"]), 3),
                "duration": round(float(e["end"] - e["start"]), 3),
                "reference": ref,
                "hypothesis": hyp,
                "is_empty_hypothesis": hyp == "",
                "blank_fraction": round(float(blank_frac), 6),
                "ref_all_in_vocab": has_only_in_vocab_chars(ref, processor),
                "audio_loaded_ok": loaded_ok,
            })

    # ------------------------------------------------------------------ #
    # Metrics                                                            #
    # ------------------------------------------------------------------ #
    pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    if pairs:
        pred_list, ref_list = zip(*pairs)
        wer = wer_metric.compute(predictions=list(pred_list), references=list(ref_list))
        cer = cer_metric.compute(predictions=list(pred_list), references=list(ref_list))
    else:
        wer, cer = 1.0, 1.0

    fair_pairs = [
        (u["hypothesis"], u["reference"])
        for u in per_utt
        if u["ref_all_in_vocab"] and u["reference"].strip()
    ]
    if fair_pairs:
        fair_preds, fair_refs = zip(*fair_pairs)
        fair_wer = wer_metric.compute(predictions=list(fair_preds), references=list(fair_refs))
        fair_cer = cer_metric.compute(predictions=list(fair_preds), references=list(fair_refs))
    else:
        fair_wer, fair_cer = float("nan"), float("nan")

    blank_fracs_all = [u["blank_fraction"] for u in per_utt]
    avg_blank_frac = float(np.mean(blank_fracs_all)) if blank_fracs_all else float("nan")
    high_blank_count = sum(x >= 0.95 for x in blank_fracs_all)

    # ------------------------------------------------------------------ #
    # Report                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("[results]")
    print(f"  utterances evaluated        : {len(per_utt)}")
    print(f"  load failures               : {load_failures}")
    print(f"  WER                         : {wer:.4f}")
    print(f"  CER                         : {cer:.4f}")
    print(f"  in-vocab subset utts        : {len(fair_pairs)} / {len(per_utt)}")
    print(f"  fair-subset WER             : {fair_wer:.4f}")
    print(f"  fair-subset CER             : {fair_cer:.4f}")
    print(f"  empty hypotheses            : {empty_count} / {len(per_utt)} ({100.0 * empty_count / max(1, len(per_utt)):.2f}%)")
    print(f"  avg blank fraction          : {avg_blank_frac:.4f}")
    print(f"  utts with blank_frac >= .95 : {high_blank_count} / {len(per_utt)} ({100.0 * high_blank_count / max(1, len(per_utt)):.2f}%)")
    print(f"  ref char OOV pct            : {cov['oov_pct']:.2f}%")
    print("=" * 72)

    # show worst blank-dominant examples
    print("\n[examples] Top 10 most blank-dominant utterances")
    for u in sorted(per_utt, key=lambda x: x["blank_fraction"], reverse=True)[:10]:
        print(f"utt={u['utt_id']} dur={u['duration']:.2f}s blank={u['blank_fraction']:.4f}")
        print(f"  REF : {u['reference']!r}")
        print(f"  HYP : {u['hypothesis']!r}")

    print("\n[examples] Random sample predictions")
    rng = np.random.default_rng(0)
    sample_n = min(8, len(per_utt))
    for idx in rng.choice(len(per_utt), size=sample_n, replace=False):
        u = per_utt[int(idx)]
        print(f"utt={u['utt_id']} dur={u['duration']:.2f}s blank={u['blank_fraction']:.4f}")
        print(f"  REF : {u['reference']!r}")
        print(f"  HYP : {u['hypothesis']!r}")

    # ------------------------------------------------------------------ #
    # Save                                                               #
    # ------------------------------------------------------------------ #
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
        summary = {
            "_summary": True,
            "n_utts": len(per_utt),
            "load_failures": load_failures,
            "wer": float(wer),
            "cer": float(cer),
            "fair_wer": float(fair_wer) if not np.isnan(fair_wer) else None,
            "fair_cer": float(fair_cer) if not np.isnan(fair_cer) else None,
            "empty_hypotheses": empty_count,
            "empty_hypothesis_pct": 100.0 * empty_count / max(1, len(per_utt)),
            "avg_blank_fraction": avg_blank_frac,
            "high_blank_frac_utts": high_blank_count,
            "oov_pct": float(cov["oov_pct"]),
            "oov_breakdown": cov["oov_breakdown"],
        }

        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for row in per_utt:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        print(f"\n[save] Wrote predictions to: {args.out_jsonl}")


if __name__ == "__main__":
    main()