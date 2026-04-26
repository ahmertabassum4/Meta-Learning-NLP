import os
import re
import json
import random
import argparse
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Union

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from evaluate import load


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
# Keep: Bengali letters/marks/digits, Latin letters, ASCII digits, space.
# Drop: punctuation, quotes, parentheses, dashes, etc.
# This dramatically shrinks the vocab and reduces rare-character noise,
# which is critical when training an XLS-R LM head from scratch on tiny data.

_KEEP_PATTERN = re.compile(
    r"[^"
    r"\u0980-\u09FF"   # Bengali block
    r"A-Za-z"          # Latin letters
    r"0-9"             # digits
    r" "               # space
    r"]"
)


def clean_text(text: str) -> str:
    text = str(text)
    # Normalize unicode (NFC) so composed/decomposed forms match
    text = unicodedata.normalize("NFC", text)
    # Drop markdown-ish junk
    text = text.replace("**", "").replace("__", "")
    # Lowercase Latin portion to shrink vocab (Bengali has no case)
    text = text.lower()
    # Replace non-kept chars with space
    text = _KEEP_PATTERN.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LocalCSFleursDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, split_dir: str, processor: Wav2Vec2Processor):
        self.df = dataframe.reset_index(drop=True)
        self.split_dir = split_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        rel_audio_path = str(item["audio"]).replace("\\", "/")
        audio_path = os.path.join(self.split_dir, rel_audio_path)

        try:
            audio, _ = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(16000, dtype=np.float32)

        text = clean_text(item["text"])

        input_values = self.processor(
            audio,
            sampling_rate=16000,
        ).input_values[0]

        labels = self.processor.tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
            "input_length": len(input_values),  # for group_by_length
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------
def build_vocab(texts: List[str]) -> Dict[str, int]:
    all_chars = set()
    for text in texts:
        all_chars.update(clean_text(text))

    # Space becomes the word delimiter "|"
    all_chars.discard(" ")

    vocab_list = sorted(all_chars)
    vocab_dict = {c: i for i, c in enumerate(vocab_list)}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics_factory(processor: Wav2Vec2Processor):
    wer_metric = load("wer")
    cer_metric = load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        # Avoid empty reference crashes in wer/cer metric
        pred_str = [clean_text(x) if clean_text(x) else " " for x in pred_str]
        label_str = [clean_text(x) if clean_text(x) else " " for x in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def split_train_valid(df: pd.DataFrame, valid_ratio: float = 0.1, seed: int = 42):
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)

    valid_size = max(1, int(len(df) * valid_ratio))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    return train_df, valid_df


def print_dataset_samples(train_df, valid_df, test_df):
    print("\nSample training texts:")
    for x in train_df["text"].head(3).tolist():
        print("-", repr(clean_text(x)))

    print("\nSample validation texts:")
    for x in valid_df["text"].head(3).tolist():
        print("-", repr(clean_text(x)))

    print("\nSample test texts:")
    for x in test_df["text"].head(3).tolist():
        print("-", repr(clean_text(x)))


# ---------------------------------------------------------------------------
# Sanity checks (catch pipeline bugs before a long training run)
# ---------------------------------------------------------------------------
def sanity_check_tokenizer(processor: Wav2Vec2Processor, sample_text: str):
    cleaned = clean_text(sample_text)
    ids = processor.tokenizer(cleaned, add_special_tokens=False).input_ids
    decoded = processor.tokenizer.decode(ids).replace(
        processor.tokenizer.pad_token, ""
    ).strip()
    # The decoder uses "|" for space internally but should render as space.
    print("\n[Tokenizer round-trip check]")
    print(f"  Original: {cleaned!r}")
    print(f"  Decoded : {decoded!r}")
    print(f"  Match   : {cleaned.replace(' ', '') == decoded.replace(' ', '')}")


def debug_model_predictions(model, processor, dataset, device, num_samples=3):
    """Print what the model actually predicts — catches blank-collapse."""
    model.eval()
    print("\n[Prediction debug]")
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            iv = torch.tensor(sample["input_values"]).unsqueeze(0).to(device)
            logits = model(iv).logits
            pred_ids = torch.argmax(logits, dim=-1)
            unique_ids = torch.unique(pred_ids).tolist()
            pred_text = processor.batch_decode(pred_ids)[0]
            label_ids = np.array(sample["labels"])
            ref_text = processor.tokenizer.decode(label_ids)
            print(f"  Sample {i}:")
            print(f"    Unique predicted token ids: {unique_ids}")
            print(f"    Pred: {pred_text[:150]!r}")
            print(f"    Ref : {ref_text[:150]!r}")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--data_dir", type=str, default="data/cs_fleurs_ben_eng_300_49")
    parser.add_argument("--output_dir", type=str, default="runs/w2v2_csfleurs_ben_eng_fix2")
    parser.add_argument("--num_train_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="0 disables early stopping. Recommended for tiny data.")
    parser.add_argument("--freeze_feature_encoder", action="store_true", default=True)
    parser.add_argument("--mask_time_prob", type=float, default=0.0,
                        help="Keep 0 while the LM head is learning; raise to 0.05 later.")
    parser.add_argument("--layerdrop", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--hidden_dropout", type=float, default=0.0)
    parser.add_argument("--feat_proj_dropout", type=float, default=0.0)
    parser.add_argument("--overfit_check", action="store_true",
                        help="Train on 4 samples for 200 steps to verify pipeline.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    train_manifest = os.path.join(train_dir, "manifest.csv")
    test_manifest = os.path.join(test_dir, "manifest.csv")

    if not os.path.exists(train_manifest):
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
    if not os.path.exists(test_manifest):
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

    full_train_df = pd.read_csv(train_manifest)
    test_df = pd.read_csv(test_manifest)

    print(f"Original training samples: {len(full_train_df)}")
    print(f"Final test samples: {len(test_df)}")

    train_df, valid_df = split_train_valid(
        full_train_df, valid_ratio=args.valid_ratio, seed=args.seed
    )

    # Overfit-mode: shrink everything to 4 samples.
    # We deliberately use AGGRESSIVE settings here: high LR, unfrozen everything,
    # zero regularization. The only question we're asking is "can the model
    # memorize 4 samples?" If it can't even do that, the pipeline is broken.
    if args.overfit_check:
        print("\n*** OVERFIT CHECK MODE: training on 4 samples ***")
        train_df = train_df.head(4).reset_index(drop=True)
        valid_df = train_df.copy()
        args.num_train_epochs = 300
        args.learning_rate = 5e-4        # much higher, cold head needs push
        args.warmup_ratio = 0.03         # tiny warmup
        args.freeze_feature_encoder = False  # let everything move
        args.batch_size = 4
        args.grad_accum = 1              # real gradient steps every batch

    print(f"Train split samples: {len(train_df)}")
    print(f"Validation split samples: {len(valid_df)}")

    print_dataset_samples(train_df, valid_df, test_df)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    valid_df.to_csv(os.path.join(args.output_dir, "valid_split.csv"), index=False)

    # Build vocab from cleaned text across all splits
    all_texts = (
        train_df["text"].astype(str).tolist()
        + valid_df["text"].astype(str).tolist()
        + test_df["text"].astype(str).tolist()
    )
    vocab_dict = build_vocab(all_texts)
    print(f"\nVocab size (incl. special tokens): {len(vocab_dict)}")
    print(f"Vocab chars: {sorted(c for c in vocab_dict if c not in ('[UNK]', '[PAD]', '|'))}")

    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
        do_lower_case=False,
    )
    # Force-clear any BOS/EOS that the constructor may have set anyway.
    # If left set, the Trainer will align the model config to them and CTC breaks.
    tokenizer.bos_token = None
    tokenizer.eos_token = None
    tokenizer._bos_token = None
    tokenizer._eos_token = None

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(args.output_dir)

    # Tokenizer sanity check
    sanity_check_tokenizer(processor, train_df["text"].iloc[0])

    train_dataset = LocalCSFleursDataset(train_df, train_dir, processor)
    valid_dataset = LocalCSFleursDataset(valid_df, train_dir, processor)
    test_dataset = LocalCSFleursDataset(test_df, test_dir, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,  # prevents inf loss from blocking gradients
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=0.0,
        layerdrop=args.layerdrop,
    )

    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = None
    model.config.eos_token_id = None

    # ------------------------------------------------------------------
    # CTC feasibility check: output_frames must be >= label_length.
    # wav2vec2 downsamples audio ~320x. If labels are too long relative
    # to audio, CTC loss is infinite/NaN and the model can never learn.
    # ------------------------------------------------------------------
    print("\n[CTC feasibility check]")
    problems = 0
    for i in range(min(len(train_dataset), 10)):
        sample = train_dataset[i]
        audio_len = len(sample["input_values"])
        label_len = len(sample["labels"])
        # XLS-R downsample factor is 320
        est_output_frames = audio_len // 320
        status = "OK" if est_output_frames >= label_len else "TOO SHORT"
        if status == "TOO SHORT":
            problems += 1
        print(f"  Sample {i}: audio={audio_len} samples (~{audio_len/16000:.2f}s), "
              f"output_frames~{est_output_frames}, label_len={label_len} [{status}]")
    if problems:
        print(f"  WARNING: {problems} samples have label_len > output_frames. "
              f"These produce inf CTC loss and prevent learning.")
    else:
        print("  All checked samples are CTC-feasible.")

    # Use fp32 for overfit check — fp16 can cause NaN with a cold CTC head
    use_fp16 = (args.device == "cuda") and (not args.overfit_check)

    # ------------------------------------------------------------------
    # Direct forward-pass sanity check: compute CTC loss on one batch and
    # verify it is finite. Also print logit stats to confirm the head
    # isn't already saturated on pad.
    # ------------------------------------------------------------------
    print("\n[Direct forward-pass CTC loss check]")
    model.to(args.device)
    model.train()
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    _batch = collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])
    _batch = {k: v.to(args.device) for k, v in _batch.items()}
    with torch.no_grad():
        _out = model(**_batch)
    print(f"  Loss        : {_out.loss.item():.4f}  "
          f"({'FINITE' if torch.isfinite(_out.loss) else 'NaN/Inf — BUG'})")
    print(f"  Logits shape: {tuple(_out.logits.shape)}  "
          f"(batch, time_frames, vocab_size)")
    print(f"  Vocab size  : {len(processor.tokenizer)}")
    # Most-predicted id per frame, summed across batch — should not be dominated
    # by pad token before training.
    _pred_ids = _out.logits.argmax(dim=-1)
    _unique, _counts = torch.unique(_pred_ids, return_counts=True)
    _pad_id = processor.tokenizer.pad_token_id
    print(f"  Pad token id: {_pad_id}")
    print(f"  Top-5 predicted ids: "
          f"{sorted(zip(_counts.tolist(), _unique.tolist()), reverse=True)[:5]}")
    print(f"  Tokenizer bos_token_id: {processor.tokenizer.bos_token_id}")
    print(f"  Tokenizer eos_token_id: {processor.tokenizer.eos_token_id}")
    print(f"  Model config bos: {model.config.bos_token_id}, eos: {model.config.eos_token_id}")
    del _batch, _out, _pred_ids

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        length_column_name="input_length",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        dataloader_num_workers=0,
        fp16=use_fp16,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=processor,
        callbacks=callbacks,
        compute_metrics=compute_metrics_factory(processor),
    )

    # Pre-training prediction check — should show mostly-pad-only output.
    # This is just to confirm the pipeline runs end-to-end before training.
    device = next(model.parameters()).device
    debug_model_predictions(model, processor, valid_dataset, device, num_samples=2)

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Post-training prediction check
    debug_model_predictions(model, processor, valid_dataset, device, num_samples=3)

    print("\nRunning final evaluation on untouched test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    test_metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
Recommended workflow:

STEP 1 — Sanity check (takes ~1-2 min). Confirms pipeline is not broken.
  python ahmer/finetune_ben-eng.py \
    --data_dir data/cs_fleurs_ben_eng_300_49 \
    --output_dir runs/sanity \
    --overfit_check \
    --device cuda

  Expect: train loss drops sharply (< 1.0), WER on those same 4 samples
  approaches 0. If it doesn't, the pipeline itself is broken.

STEP 2 — Full run:
  python ahmer/finetune_ben-eng.py \
    --data_dir data/cs_fleurs_ben_eng_300_49 \
    --output_dir runs/w2v2_csfleurs_ben_eng_fix2 \
    --num_train_epochs 20 \
    --batch_size 4 \
    --grad_accum 2 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --freeze_feature_encoder \
    --device cuda

STEP 3 (optional) — once loss is clearly decreasing and CER < 0.9, add
regularization for a second run:
    --mask_time_prob 0.05 --layerdrop 0.05 --hidden_dropout 0.1
"""

# import os
# import json
# import random
# import torch
# import librosa
# import numpy as np
# import pandas as pd
# import argparse
# from typing import List, Dict, Union
# from dataclasses import dataclass
# from torch.utils.data import Dataset
# from transformers import (
#     Wav2Vec2ForCTC,
#     Wav2Vec2Processor,
#     Wav2Vec2CTCTokenizer,
#     Wav2Vec2FeatureExtractor,
#     Trainer,
#     TrainingArguments,
#     EarlyStoppingCallback,
# )
# from evaluate import load


# def clean_text(text: str) -> str:
#     text = str(text).replace("**", "").replace("__", "").strip()
#     return " ".join(text.split())


# class LocalCSFleursDataset(Dataset):
#     def __init__(self, dataframe: pd.DataFrame, split_dir: str, processor: Wav2Vec2Processor):
#         self.df = dataframe.reset_index(drop=True)
#         self.split_dir = split_dir
#         self.processor = processor

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         item = self.df.iloc[idx]

#         rel_audio_path = str(item["audio"]).replace("\\", "/")
#         audio_path = os.path.join(self.split_dir, rel_audio_path)

#         try:
#             audio, _ = librosa.load(audio_path, sr=16000)
#         except Exception as e:
#             print(f"Error loading {audio_path}: {e}")
#             audio = np.zeros(16000, dtype=np.float32)

#         text = clean_text(item["text"])
#         input_values = self.processor(audio, sampling_rate=16000).input_values[0]
#         labels = self.processor.tokenizer(text).input_ids

#         return {
#             "input_values": input_values,
#             "labels": labels,
#         }


# @dataclass
# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             return_tensors="pt",
#         )

#         labels_batch = self.processor.tokenizer.pad(
#             label_features,
#             padding=self.padding,
#             return_tensors="pt",
#         )

#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch["labels"] = labels
#         return batch


# def build_vocab(texts: List[str]) -> Dict[str, int]:
#     all_chars = set()
#     for text in texts:
#         all_chars.update(clean_text(text))

#     vocab_list = sorted(list(all_chars))
#     vocab_dict = {c: i for i, c in enumerate(vocab_list)}

#     if " " in vocab_dict:
#         space_idx = vocab_dict[" "]
#         del vocab_dict[" "]
#         vocab_dict["|"] = space_idx
#     else:
#         vocab_dict["|"] = len(vocab_dict)

#     vocab_dict["[UNK]"] = len(vocab_dict)
#     vocab_dict["[PAD]"] = len(vocab_dict)

#     return vocab_dict


# def compute_metrics_factory(processor: Wav2Vec2Processor):
#     wer_metric = load("wer")
#     cer_metric = load("cer")

#     def compute_metrics(pred):
#         pred_logits = pred.predictions
#         pred_ids = np.argmax(pred_logits, axis=-1)

#         pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#         pred_str = processor.batch_decode(pred_ids)
#         label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#         wer = wer_metric.compute(predictions=pred_str, references=label_str)
#         cer = cer_metric.compute(predictions=pred_str, references=label_str)

#         return {
#             "wer": wer,
#             "cer": cer,
#         }

#     return compute_metrics


# def split_train_valid(df: pd.DataFrame, valid_ratio: float = 0.1, seed: int = 42):
#     indices = list(range(len(df)))
#     random.Random(seed).shuffle(indices)

#     valid_size = max(1, int(len(df) * valid_ratio))
#     valid_indices = indices[:valid_size]
#     train_indices = indices[valid_size:]

#     train_df = df.iloc[train_indices].reset_index(drop=True)
#     valid_df = df.iloc[valid_indices].reset_index(drop=True)

#     return train_df, valid_df


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-xls-r-300m")
#     parser.add_argument("--data_dir", type=str, default="data/cs_fleurs_ben_eng_300_49")
#     parser.add_argument("--output_dir", type=str, default="runs/w2v2_csfleurs_ben_eng")
#     parser.add_argument("--num_train_epochs", type=int, default=12)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--grad_accum", type=int, default=2)
#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--valid_ratio", type=float, default=0.1)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--early_stopping_patience", type=int, default=3)
#     parser.add_argument("--freeze_feature_encoder", action="store_true")
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#         choices=["cpu", "cuda"],
#     )
#     args = parser.parse_args()

#     print(f"Using device: {args.device}")

#     train_dir = os.path.join(args.data_dir, "train")
#     test_dir = os.path.join(args.data_dir, "test")

#     train_manifest = os.path.join(train_dir, "manifest.csv")
#     test_manifest = os.path.join(test_dir, "manifest.csv")

#     if not os.path.exists(train_manifest):
#         raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
#     if not os.path.exists(test_manifest):
#         raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

#     full_train_df = pd.read_csv(train_manifest)
#     test_df = pd.read_csv(test_manifest)

#     print(f"Original training samples: {len(full_train_df)}")
#     print(f"Final test samples: {len(test_df)}")

#     train_df, valid_df = split_train_valid(
#         full_train_df,
#         valid_ratio=args.valid_ratio,
#         seed=args.seed,
#     )

#     print(f"Train split samples: {len(train_df)}")
#     print(f"Validation split samples: {len(valid_df)}")

#     os.makedirs(args.output_dir, exist_ok=True)

#     train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
#     valid_df.to_csv(os.path.join(args.output_dir, "valid_split.csv"), index=False)

#     all_texts = (
#         train_df["text"].astype(str).tolist()
#         + valid_df["text"].astype(str).tolist()
#         + test_df["text"].astype(str).tolist()
#     )

#     vocab_dict = build_vocab(all_texts)
#     vocab_path = os.path.join(args.output_dir, "vocab.json")
#     with open(vocab_path, "w", encoding="utf-8") as f:
#         json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

#     tokenizer = Wav2Vec2CTCTokenizer(
#         vocab_path,
#         unk_token="[UNK]",
#         pad_token="[PAD]",
#         word_delimiter_token="|",
#     )

#     feature_extractor = Wav2Vec2FeatureExtractor(
#         feature_size=1,
#         sampling_rate=16000,
#         padding_value=0.0,
#         do_normalize=True,
#         return_attention_mask=True,
#     )

#     processor = Wav2Vec2Processor(
#         feature_extractor=feature_extractor,
#         tokenizer=tokenizer,
#     )
#     processor.save_pretrained(args.output_dir)

#     train_dataset = LocalCSFleursDataset(train_df, train_dir, processor)
#     valid_dataset = LocalCSFleursDataset(valid_df, train_dir, processor)
#     test_dataset = LocalCSFleursDataset(test_df, test_dir, processor)

#     model = Wav2Vec2ForCTC.from_pretrained(
#         args.base_model,
#         ctc_loss_reduction="mean",
#         pad_token_id=processor.tokenizer.pad_token_id,
#         vocab_size=len(processor.tokenizer),
#         attention_dropout=0.1,
#         hidden_dropout=0.1,
#         feat_proj_dropout=0.0,
#         mask_time_prob=0.05,
#         layerdrop=0.05,
#     )

#     if args.freeze_feature_encoder:
#         model.freeze_feature_encoder()

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         group_by_length=True,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         logging_strategy="steps",
#         logging_steps=5,
#         num_train_epochs=args.num_train_epochs,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         warmup_steps=25,
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         dataloader_num_workers=0,
#         fp16=(args.device == "cuda"),
#         push_to_hub=False,
#         report_to="none",
#         remove_unused_columns=False,
#         seed=args.seed,
#     )

#     callbacks = []
#     if args.early_stopping_patience > 0:
#         callbacks.append(
#             EarlyStoppingCallback(
#                 early_stopping_patience=args.early_stopping_patience
#             )
#         )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         processing_class=processor,
#         callbacks=callbacks,
#         compute_metrics=compute_metrics_factory(processor),
#     )

#     print("Starting training with validation split from training data...")
#     trainer.train()

#     print(f"Saving final model to {args.output_dir}")
#     trainer.save_model(args.output_dir)
#     processor.save_pretrained(args.output_dir)

#     print("Running final evaluation on untouched test set...")
#     test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

#     test_metrics_path = os.path.join(args.output_dir, "test_metrics.json")
#     with open(test_metrics_path, "w", encoding="utf-8") as f:
#         json.dump(test_metrics, f, indent=2)

#     print("Final test metrics:")
#     print(json.dumps(test_metrics, indent=2))


# if __name__ == "__main__":
#     main()

# """
# Example:
# python ahmer/finetune_ben-eng.py \
#   --data_dir data/cs_fleurs_ben_eng_300_49 \
#   --output_dir runs/w2v2_csfleurs_ben_eng_clean \
#   --num_train_epochs 12 \
#   --batch_size 4 \
#   --grad_accum 2 \
#   --learning_rate 1e-4 \
#   --freeze_feature_encoder \
#   --device cuda
# """


# import os
# import json
# import torch
# import librosa
# import numpy as np
# import pandas as pd
# import argparse
# from typing import List, Dict, Union
# from dataclasses import dataclass
# from torch.utils.data import Dataset
# from transformers import (
#     Wav2Vec2ForCTC,
#     Wav2Vec2Processor,
#     Wav2Vec2CTCTokenizer,
#     Wav2Vec2FeatureExtractor,
#     Trainer,
#     TrainingArguments,
#     EarlyStoppingCallback,
# )

# def clean_text(text: str) -> str:
#     text = str(text).replace("**", "").replace("__", "").strip()
#     return " ".join(text.split())

# class LocalCSFleursDataset(Dataset):
#     def __init__(self, manifest_path: str, split_dir: str, processor: Wav2Vec2Processor):
#         self.df = pd.read_csv(manifest_path)
#         self.split_dir = split_dir
#         self.processor = processor

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         item = self.df.iloc[idx]

#         rel_audio_path = str(item["audio"]).replace("\\", "/")
#         audio_path = os.path.join(self.split_dir, rel_audio_path)

#         try:
#             audio, _ = librosa.load(audio_path, sr=16000)
#         except Exception as e:
#             print(f"Error loading {audio_path}: {e}")
#             audio = np.zeros(16000, dtype=np.float32)

#         text = clean_text(item["text"])

#         input_values = self.processor(audio, sampling_rate=16000).input_values[0]
#         labels = self.processor.tokenizer(text).input_ids

#         return {
#             "input_values": input_values,
#             "labels": labels
#         }

# @dataclass
# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             return_tensors="pt",
#         )

#         labels_batch = self.processor.tokenizer.pad(
#             label_features,
#             padding=self.padding,
#             return_tensors="pt",
#         )

#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch["labels"] = labels
#         return batch

# def build_vocab(texts: List[str]) -> Dict[str, int]:
#     all_chars = set()
#     for text in texts:
#         all_chars.update(clean_text(text))

#     vocab_list = sorted(list(all_chars))
#     vocab_dict = {c: i for i, c in enumerate(vocab_list)}

#     # Replace space with word delimiter
#     if " " in vocab_dict:
#         space_index = vocab_dict[" "]
#         del vocab_dict[" "]
#         vocab_dict["|"] = space_index
#     else:
#         vocab_dict["|"] = len(vocab_dict)

#     vocab_dict["[UNK]"] = len(vocab_dict)
#     vocab_dict["[PAD]"] = len(vocab_dict)

#     return vocab_dict

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-xls-r-300m")
#     parser.add_argument("--data_dir", type=str, default="data/cs_fleurs_ben_eng_300_49")
#     parser.add_argument("--output_dir", type=str, default="runs/w2v2_csfleurs_ben_eng")
#     parser.add_argument("--num_train_epochs", type=int, default=12)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--grad_accum", type=int, default=2)
#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--early_stopping_patience", type=int, default=3)
#     parser.add_argument("--freeze_feature_encoder", action="store_true")
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#         choices=["cpu", "cuda"],
#         help="Device to use"
#     )
#     args = parser.parse_args()

#     print(f"Using device: {args.device}")

#     train_dir = os.path.join(args.data_dir, "train")
#     test_dir = os.path.join(args.data_dir, "test")

#     train_manifest = os.path.join(train_dir, "manifest.csv")
#     test_manifest = os.path.join(test_dir, "manifest.csv")

#     if not os.path.exists(train_manifest):
#         raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
#     if not os.path.exists(test_manifest):
#         raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

#     train_df = pd.read_csv(train_manifest)
#     test_df = pd.read_csv(test_manifest)

#     print(f"Train samples: {len(train_df)}")
#     print(f"Test samples: {len(test_df)}")

#     if "duration" in train_df.columns:
#         train_hours = pd.to_numeric(train_df["duration"], errors="coerce").fillna(0).sum() / 3600
#         print(f"Train duration: {train_hours:.2f} hours")
#     if "duration" in test_df.columns:
#         test_hours = pd.to_numeric(test_df["duration"], errors="coerce").fillna(0).sum() / 3600
#         print(f"Test duration: {test_hours:.2f} hours")

#     all_texts = train_df["text"].astype(str).tolist() + test_df["text"].astype(str).tolist()

#     os.makedirs(args.output_dir, exist_ok=True)

#     vocab_dict = build_vocab(all_texts)
#     vocab_path = os.path.join(args.output_dir, "vocab.json")
#     with open(vocab_path, "w", encoding="utf-8") as f:
#         json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

#     tokenizer = Wav2Vec2CTCTokenizer(
#         vocab_path,
#         unk_token="[UNK]",
#         pad_token="[PAD]",
#         word_delimiter_token="|"
#     )
#     feature_extractor = Wav2Vec2FeatureExtractor(
#         feature_size=1,
#         sampling_rate=16000,
#         padding_value=0.0,
#         do_normalize=True,
#         return_attention_mask=True
#     )
#     processor = Wav2Vec2Processor(
#         feature_extractor=feature_extractor,
#         tokenizer=tokenizer
#     )
#     processor.save_pretrained(args.output_dir)

#     train_dataset = LocalCSFleursDataset(train_manifest, train_dir, processor)
#     eval_dataset = LocalCSFleursDataset(test_manifest, test_dir, processor)

#     model = Wav2Vec2ForCTC.from_pretrained(
#         args.base_model,
#         ctc_loss_reduction="mean",
#         pad_token_id=processor.tokenizer.pad_token_id,
#         vocab_size=len(processor.tokenizer),
#         attention_dropout=0.1,
#         hidden_dropout=0.1,
#         feat_proj_dropout=0.0,
#         mask_time_prob=0.05,
#         layerdrop=0.05,
#     )

#     if args.freeze_feature_encoder:
#         model.freeze_feature_encoder()

#     print("Model loaded successfully.")

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         group_by_length=True,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         logging_strategy="steps",
#         logging_steps=5,
#         num_train_epochs=args.num_train_epochs,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         warmup_ratio=0.1,
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         dataloader_num_workers=0,
#         fp16=(args.device == "cuda"),
#         push_to_hub=False,
#         report_to="none",
#         remove_unused_columns=False,
#     )

#     callbacks = []
#     if args.early_stopping_patience > 0:
#         callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         processing_class=processor,
#         callbacks=callbacks,
#     )

#     print("Starting training...")
#     trainer.train()

#     print(f"Saving final model to {args.output_dir}")
#     trainer.save_model(args.output_dir)
#     processor.save_pretrained(args.output_dir)

# if __name__ == "__main__":
#     main()

# """
# Example:
# python ahmer/finetune_ben-eng.py \
#   --data_dir data/cs_fleurs_ben_eng_300_49 \
#   --output_dir runs/w2v2_csfleurs_ben_eng \
#   --num_train_epochs 12 \
#   --batch_size 4 \
#   --grad_accum 2 \
#   --learning_rate 1e-4 \
#   --freeze_feature_encoder \
#   --device cuda
# """