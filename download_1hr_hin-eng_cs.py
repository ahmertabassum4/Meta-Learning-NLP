#!/usr/bin/env python3
"""
Download ~1 hour of Hindi-English code-switched ASR data from OpenSLR 104
(MUCS 2021 sub-task2 train set), with all metadata needed for fine-tuning.

Source: https://www.openslr.org/104/
Full train set: 7.3 GB / 89.86 hours.

Known archive layout (confirmed from the test set):
    train-SLR-hin/
        transcripts/
            segments      (utt-id  recording-id  start  end)
            text          (utt-id  <transcript>)
            wav.scp       (recording-id  <path-or-pipe-to-wav>)
            utt2spk       (utt-id  speaker-id)
            spk2utt
            spkr_list
        <recording-id>.wav  ... many .wav files at the dataset root

Strategy (no full local copy of the 7.3 GB tarball):
    PASS 1: stream the tarball and extract ONLY the small `transcripts/` dir.
    SELECT: parse segments, pick a random subset whose total duration ~= 1 hr.
    PASS 2: re-stream the tarball; extract ONLY the .wav files we need.
    VERIFY: drop any utterance missing audio or transcript; write final manifest.

Output (ready for HuggingFace / Whisper / wav2vec2 fine-tuning):
    out/
      audio/                  extracted .wav files (16 kHz, 16-bit)
      text                    filtered: <utt-id> <transcript>
      segments                filtered: <utt> <rec> <start> <end>
      utt2spk                 filtered (if present)
      wav.scp                 rewritten to local paths in audio/
      manifest.jsonl          one JSON per utterance: {audio, text, duration, ...}
      _meta_full/             original transcripts/ dir preserved
      README.txt

Requires: Python 3.8+, stdlib only.
Usage:    python download_1hr_hineng_cs.py --out ./hineng_cs_1hr --hours 1.0
"""

import argparse
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz",
    "https://openslr.elda.org/resources/104/Hindi-English_train.tar.gz",
    "https://openslr.magicdatatech.com/resources/104/Hindi-English_train.tar.gz",
]


# --------------------------- HTTP helpers ---------------------------

def open_remote_stream(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "openslr-1hr-fetch/2.0"})
    return urllib.request.urlopen(req, timeout=120)


def try_mirrors():
    last_err = None
    for url in MIRRORS:
        try:
            print(f"[net] connecting: {url}", file=sys.stderr)
            resp = open_remote_stream(url)
            print(f"[net] connected.", file=sys.stderr)
            return url, resp
        except Exception as e:
            print(f"[net] mirror failed ({e})", file=sys.stderr)
            last_err = e
    raise RuntimeError(f"All mirrors failed. Last error: {last_err}")


# --------------------------- PASS 1: metadata ---------------------------

def pass1_extract_transcripts_dir(meta_root: Path):
    """
    Stream the tarball; extract every file under any '*/transcripts/' folder.
    Returns dict basename -> local Path (e.g. 'segments' -> Path).
    Also returns the dataset root prefix inside the archive (e.g. 'train-SLR-hin').
    """
    meta_root.mkdir(parents=True, exist_ok=True)
    saved = {}
    archive_root = None  # e.g. "train-SLR-hin"

    _, resp = try_mirrors()
    try:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                # Match anything inside a "transcripts/" directory.
                parts = Path(member.name).parts
                if "transcripts" not in parts:
                    continue

                # Capture the dataset root (first path segment).
                if archive_root is None and len(parts) >= 1:
                    archive_root = parts[0]

                f = tf.extractfile(member)
                if f is None:
                    continue
                target = meta_root / Path(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f.read())
                saved[Path(member.name).name] = target
                print(f"[meta] {member.name} ({member.size} bytes)",
                      file=sys.stderr)
    finally:
        resp.close()

    if not saved:
        sys.exit("[error] No 'transcripts/' files found in archive — "
                 "layout may have changed.")
    print(f"[pass1] saved {len(saved)} metadata files; "
          f"archive root = {archive_root!r}", file=sys.stderr)
    return saved, archive_root


# --------------------------- parsers ---------------------------

def parse_segments(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            utt, rec, s, e = parts
            try:
                s, e = float(s), float(e)
            except ValueError:
                continue
            if e > s:
                rows.append((utt, rec, s, e, e - s))
    return rows


def parse_text(path):
    utt2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            utt2text[parts[0]] = parts[1] if len(parts) == 2 else ""
    return utt2text


def parse_utt2spk(path):
    out = {}
    if path is None or not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def parse_wav_scp(path):
    """recording-id -> a .wav filename (basename only).
       Handles both plain paths and Kaldi pipe entries."""
    rec2wav = {}
    wav_re = re.compile(r"([^\s/\\]+\.wav)", re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            rec_id, rest = parts
            m = wav_re.search(rest)
            if m:
                rec2wav[rec_id] = m.group(1)
    return rec2wav


# --------------------------- selection ---------------------------

def select_utterances(segments, target_seconds, seed):
    """
    Pre-filter to a sensible utterance length range (1s..30s) which is what
    fine-tuning recipes (Whisper, wav2vec2) actually want, then shuffle and
    greedily pick until total duration >= target.
    """
    usable = [r for r in segments if 1.0 <= r[4] <= 30.0]
    rng = random.Random(seed)
    rng.shuffle(usable)
    chosen, total = [], 0.0
    for row in usable:
        if total >= target_seconds:
            break
        chosen.append(row)
        total += row[4]
    return chosen, total


# --------------------------- PASS 2: audio ---------------------------

def pass2_extract_audio(out_audio_dir: Path, needed_wav_basenames: set):
    """
    Re-stream the archive. Extract only .wav members whose basename is in
    needed_wav_basenames. Stop early once all are found.
    """
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}
    target_count = len(needed_wav_basenames)

    _, resp = try_mirrors()
    try:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if not member.name.lower().endswith(".wav"):
                    continue
                base = os.path.basename(member.name)
                if base not in needed_wav_basenames:
                    continue
                if base in extracted:
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                target = out_audio_dir / base
                target.write_bytes(f.read())
                extracted[base] = target
                print(f"[audio] {len(extracted)}/{target_count}  {base}",
                      file=sys.stderr)
                if len(extracted) >= target_count:
                    break
    finally:
        resp.close()
    return extracted


# --------------------------- writers ---------------------------

def write_outputs(out_dir, chosen, utt2text, utt2spk, rec2wav, audio_dir):
    """Write filtered text/segments/utt2spk/wav.scp + a JSONL manifest."""
    text_p = out_dir / "text"
    seg_p = out_dir / "segments"
    u2s_p = out_dir / "utt2spk"
    scp_p = out_dir / "wav.scp"
    man_p = out_dir / "manifest.jsonl"

    seen_recs = {}
    with open(text_p, "w", encoding="utf-8") as ft, \
         open(seg_p, "w", encoding="utf-8") as fs, \
         open(u2s_p, "w", encoding="utf-8") as fu, \
         open(scp_p, "w", encoding="utf-8") as fw, \
         open(man_p, "w", encoding="utf-8") as fm:

        for utt, rec, s, e, dur in chosen:
            text = utt2text[utt]
            spk = utt2spk.get(utt, utt.split("_")[0])
            wav_base = rec2wav[rec]
            wav_local = (audio_dir / wav_base).resolve()

            ft.write(f"{utt} {text}\n")
            fs.write(f"{utt} {rec} {s:.3f} {e:.3f}\n")
            fu.write(f"{utt} {spk}\n")
            if rec not in seen_recs:
                fw.write(f"{rec} {wav_local}\n")
                seen_recs[rec] = wav_local

            fm.write(json.dumps({
                "utt_id": utt,
                "recording_id": rec,
                "audio": str(wav_local),
                "text": text,
                "speaker": spk,
                "start": round(s, 3),
                "end": round(e, 3),
                "duration": round(dur, 3),
                "sampling_rate": 16000,
            }, ensure_ascii=False) + "\n")

    return text_p, seg_p, u2s_p, scp_p, man_p


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./hineng_cs_1hr",
                    help="Output directory")
    ap.add_argument("--hours", type=float, default=1.0,
                    help="Target hours of audio (default: 1.0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    audio_dir = out_dir / "audio"
    meta_root = out_dir / "_meta_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] target: {args.hours} h "
          f"({args.hours*3600:.0f} s)  out: {out_dir}", file=sys.stderr)

    # ---- Pass 1: metadata ----
    print("\n[1/4] streaming archive for transcripts/ metadata...",
          file=sys.stderr)
    saved, archive_root = pass1_extract_transcripts_dir(meta_root)

    required = ["segments", "text", "wav.scp"]
    missing = [n for n in required if n not in saved]
    if missing:
        sys.exit(f"[error] required metadata missing: {missing}")

    segs_path = saved["segments"]
    text_path = saved["text"]
    scp_path  = saved["wav.scp"]
    u2s_path  = saved.get("utt2spk")

    # ---- Selection ----
    print("\n[2/4] parsing & selecting utterances...", file=sys.stderr)
    segs   = parse_segments(segs_path)
    utt2t  = parse_text(text_path)
    rec2w  = parse_wav_scp(scp_path)
    utt2sp = parse_utt2spk(u2s_path)

    print(f"[info] segments: {len(segs)}  texts: {len(utt2t)}  "
          f"recordings: {len(rec2w)}  utt2spk: {len(utt2sp)}", file=sys.stderr)

    # Keep only utterances that have a transcript AND a wav.scp entry.
    pre_clean = [r for r in segs if r[0] in utt2t and r[1] in rec2w]
    dropped = len(segs) - len(pre_clean)
    if dropped:
        print(f"[info] dropped {dropped} utts missing text or wav mapping",
              file=sys.stderr)

    chosen, total_sec = select_utterances(pre_clean, args.hours * 3600,
                                          args.seed)
    if total_sec < 0.9 * args.hours * 3600:
        print(f"[warn] only collected {total_sec/3600:.2f}h of usable utts; "
              f"requested {args.hours}h", file=sys.stderr)
    print(f"[info] selected {len(chosen)} utts = "
          f"{total_sec/60:.2f} min ({total_sec/3600:.3f} h)", file=sys.stderr)

    needed_wavs = {rec2w[rec] for _, rec, *_ in chosen}
    print(f"[info] need {len(needed_wavs)} unique .wav files",
          file=sys.stderr)

    # ---- Pass 2: audio ----
    print("\n[3/4] streaming archive again for audio...", file=sys.stderr)
    extracted = pass2_extract_audio(audio_dir, needed_wavs)
    missing_wavs = needed_wavs - set(extracted.keys())
    if missing_wavs:
        print(f"[warn] {len(missing_wavs)} wav files NOT found in archive",
              file=sys.stderr)

    # ---- Verify ----
    print("\n[4/4] verifying & writing manifest...", file=sys.stderr)
    final = []
    for row in chosen:
        utt, rec, s, e, dur = row
        wav_base = rec2w[rec]
        if wav_base not in extracted:
            continue
        if utt not in utt2t:
            continue
        if not (audio_dir / wav_base).exists():
            continue
        final.append(row)

    final_total = sum(r[4] for r in final)
    print(f"[info] verified {len(final)} utts = "
          f"{final_total/3600:.3f} h", file=sys.stderr)

    if len(final) < 0.5 * len(chosen):
        sys.exit(f"[error] too many utts failed verification "
                 f"({len(final)}/{len(chosen)}). Aborting.")

    write_outputs(out_dir, final, utt2t, utt2sp, rec2w, audio_dir)

    (out_dir / "README.txt").write_text(
        "OpenSLR 104 - Hindi-English Code-Switched ASR (MUCS 2021)\n"
        "Source : https://www.openslr.org/104/\n"
        "License: CC BY-SA 4.0\n"
        "Domain : spoken tutorials (technical lectures, NOT spontaneous chat)\n"
        f"Subset : {args.hours} h target, seed={args.seed}\n"
        f"Final  : {len(final)} utterances, "
        f"{final_total/3600:.3f} h of audio ({final_total/60:.2f} min)\n\n"
        "Layout:\n"
        "  audio/          .wav files (16 kHz, 16-bit)\n"
        "  text            <utt-id> <transcript>\n"
        "  segments        <utt> <rec> <start_sec> <end_sec>\n"
        "  utt2spk         <utt-id> <speaker-id>\n"
        "  wav.scp         <rec-id> <local-wav-path>\n"
        "  manifest.jsonl  one JSON per utterance (HuggingFace-friendly)\n"
        "  _meta_full/     original transcripts/ from archive\n\n"
        "Note: 'segments' uses TIME OFFSETS within the recording wav.\n"
        "If your trainer needs pre-cut clips, slice each wav per-utterance\n"
        "using ffmpeg/sox/torchaudio at the start/end timestamps.\n"
    )

    print(f"\n[done] subset ready at: {out_dir}", file=sys.stderr)
    print(f"       {len(final)} utts | {final_total/3600:.3f} h",
          file=sys.stderr)


if __name__ == "__main__":
    main()