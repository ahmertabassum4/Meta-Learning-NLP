#!/usr/bin/env python3
"""
Prepare a custom 300/49 Bengali-English split from CS-FLEURS MMS-Test.

What this script does:
1. Downloads only the CS-FLEURS ben-eng MMS subset from Hugging Face
2. Reads mms/test/metadata.jsonl
3. Filters rows belonging to Bengali-English
4. Creates a custom split: 300 train / 49 test
5. Copies audio into train/ and test/ folders
6. Writes manifests for ASR experiments

Important:
- This is NOT the official CS-FLEURS split.
- Bengali-English in CS-FLEURS is test-only, so this creates a custom pseudo-train/pseudo-test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download


REPO_ID = "byan/cs-fleurs"
BEN_ENG_AUDIO_DIR = "mms/test/audio/cs_ben_eng_n1_0.3_vfilt_vconcat_vF_feb14"
METADATA_PATH = "mms/test/metadata.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cs_fleurs_ben_eng_300_49",
        help="Where to write the prepared dataset.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=300,
        help="Number of training examples.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=49,
        help="Number of test examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HF cache directory.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying audio files.",
    )
    return parser.parse_args()


def download_subset(cache_dir: str | None = None) -> Path:
    """
    Download only the ben-eng MMS files plus metadata.
    """
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[
            METADATA_PATH,
            f"{BEN_ENG_AUDIO_DIR}/*",
        ],
        cache_dir=cache_dir,
        local_dir=None,  # keep standard HF cache layout
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def read_metadata(metadata_file: Path) -> List[Dict]:
    rows = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_ben_eng_row(row: Dict) -> bool:
    """
    Robust filter for Bengali-English MMS rows.

    We rely primarily on file_name/path patterns because the dataset tree
    for MMS explicitly stores ben-eng audio in the ben-eng folder.
    """
    file_name = str(row.get("file_name", ""))
    language = str(row.get("language", "")).lower()
    text = str(row.get("text", ""))

    # Primary check: filename/path pattern
    if "cs_ben_eng" in file_name:
        return True

    # Backup checks in case metadata format differs
    if language in {"ben-eng", "ben_eng", "bengali-english", "bengali_english"}:
        return True

    # Conservative fallback: not enough signal
    _ = text
    return False


def resolve_audio_path(repo_root: Path, row: Dict) -> Path:
    """
    Resolve audio path from metadata row.
    """
    file_name = str(row["file_name"])

    # Common case: metadata stores a relative path like
    # "audio/cs_ben_eng.../xyz.wav" or "cs_ben_eng.../xyz.wav"
    candidate_paths = [
        repo_root / "mms/test" / file_name,
        repo_root / file_name,
        repo_root / BEN_ENG_AUDIO_DIR / Path(file_name).name,
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not resolve audio path for file_name={file_name!r}")


def ensure_dirs(base: Path) -> None:
    for split in ["train", "test"]:
        (base / split / "audio").mkdir(parents=True, exist_ok=True)


def write_manifest(rows: List[Dict], split_dir: Path) -> None:
    manifest_path = split_dir / "manifest.csv"
    transcripts_path = split_dir / "transcripts.txt"

    with manifest_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["id", "audio", "text", "duration", "speaker", "language"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "audio": row["local_audio_relpath"],
                    "text": row.get("text", ""),
                    "duration": row.get("duration", ""),
                    "speaker": row.get("speaker", ""),
                    "language": row.get("language", ""),
                }
            )

    with transcripts_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['id']}\t{row.get('text', '')}\n")


def materialize_split(
    rows: List[Dict],
    split_name: str,
    out_dir: Path,
    repo_root: Path,
    use_symlink: bool = False,
) -> List[Dict]:
    split_audio_dir = out_dir / split_name / "audio"
    prepared_rows = []

    for row in rows:
        src = resolve_audio_path(repo_root, row)
        ext = src.suffix if src.suffix else ".wav"
        dst = split_audio_dir / f"{row['id']}{ext}"

        if dst.exists():
            dst.unlink()

        if use_symlink:
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

        new_row = dict(row)
        new_row["local_audio_relpath"] = f"audio/{dst.name}"
        prepared_rows.append(new_row)

    write_manifest(prepared_rows, out_dir / split_name)
    return prepared_rows


def main() -> None:
    args = parse_args()
    total_needed = args.train_size + args.test_size

    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)

    print("Downloading ben-eng subset from CS-FLEURS...")
    repo_root = download_subset(cache_dir=args.cache_dir)

    metadata_file = repo_root / METADATA_PATH
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"Reading metadata: {metadata_file}")
    rows = read_metadata(metadata_file)

    ben_eng_rows = [row for row in rows if is_ben_eng_row(row)]

    # Final safety pass: only keep rows whose audio path resolves
    filtered_rows = []
    for row in ben_eng_rows:
        try:
            _ = resolve_audio_path(repo_root, row)
            filtered_rows.append(row)
        except FileNotFoundError:
            pass

    ben_eng_rows = filtered_rows

    print(f"Detected ben-eng examples: {len(ben_eng_rows)}")

    if len(ben_eng_rows) < total_needed:
        raise ValueError(
            f"Not enough ben-eng examples. Needed {total_needed}, found {len(ben_eng_rows)}."
        )

    random.seed(args.seed)
    random.shuffle(ben_eng_rows)

    selected = ben_eng_rows[:total_needed]
    train_rows = selected[: args.train_size]
    test_rows = selected[args.train_size : args.train_size + args.test_size]

    print(f"Creating train split with {len(train_rows)} examples...")
    materialize_split(
        train_rows,
        split_name="train",
        out_dir=out_dir,
        repo_root=repo_root,
        use_symlink=args.symlink,
    )

    print(f"Creating test split with {len(test_rows)} examples...")
    materialize_split(
        test_rows,
        split_name="test",
        out_dir=out_dir,
        repo_root=repo_root,
        use_symlink=args.symlink,
    )

    summary = {
        "repo_id": REPO_ID,
        "source_subset": "mms/test",
        "pair": "ben-eng",
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "seed": args.seed,
        "note": (
            "This is a custom split created from CS-FLEURS ben-eng MMS test-only data, "
            "not an official train/test split."
        ),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Output written to: {out_dir.resolve()}")
    print("Files created:")
    print(f"  {out_dir}/train/audio/")
    print(f"  {out_dir}/train/manifest.csv")
    print(f"  {out_dir}/train/transcripts.txt")
    print(f"  {out_dir}/test/audio/")
    print(f"  {out_dir}/test/manifest.csv")
    print(f"  {out_dir}/test/transcripts.txt")
    print(f"  {out_dir}/summary.json")


if __name__ == "__main__":
    main()