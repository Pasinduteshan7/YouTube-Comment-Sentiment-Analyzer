"""
Downloads the GoEmotions dataset (Google Research, ~58k Reddit comments,
28 emotion labels, multi-label) and prepares it as a clean CSV ready for
fine-tuning a multi-label emotion classifier.

Run this on your own machine — needs internet access to download from
Hugging Face. Takes about 1-2 minutes depending on connection.

Usage:
    pip install datasets pandas --break-system-packages   (if not already installed)
    python prepare_goemotions.py
"""

import pandas as pd
from datasets import load_dataset

OUTPUT_TRAIN = "goemotions_train.csv"
OUTPUT_VAL   = "goemotions_val.csv"
OUTPUT_TEST  = "goemotions_test.csv"

# Official 28-label order for the "simplified" config — index 0 to 27.
# This order matters: it must match the order used in the fine-tuning
# script's label list exactly, since the model's output neurons are
# positional, not name-based.
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]


def convert_split(split_data, split_name: str) -> pd.DataFrame:
    """
    Converts a HuggingFace GoEmotions split into a flat CSV-friendly format.
    Each row gets:
      - text: the comment
      - labels: comma-separated emotion names, e.g. "joy,gratitude"
      - one binary column per emotion (1 if present, 0 if not) —
        this is the format the fine-tuning script will train on directly.
    """
    rows = []
    dropped_unclear = 0

    for example in split_data:
        text = example["text"].strip()
        label_ids = example["labels"]  # list of integer indices into EMOTION_LABELS

        if not text or len(text) < 3:
            continue

        # GoEmotions marks some comments as "very unclear" with no labels at all —
        # these aren't useful for training and get skipped
        if len(label_ids) == 0:
            dropped_unclear += 1
            continue

        label_names = [EMOTION_LABELS[i] for i in label_ids]

        row = {
            "text": text,
            "labels": ",".join(label_names),
        }
        # one-hot binary columns — 1 if this emotion applies, 0 otherwise
        for emo in EMOTION_LABELS:
            row[emo] = 1 if emo in label_names else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {split_name}: {len(df)} usable rows ({dropped_unclear} unclear/unlabeled dropped)")
    return df


def print_label_distribution(df: pd.DataFrame):
    """Shows how many examples exist per emotion — flags the rare ones to expect weaker accuracy on."""
    print("\n  Label distribution (examples per emotion):")
    counts = {emo: int(df[emo].sum()) for emo in EMOTION_LABELS}
    for emo, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count // 200, 50)
        flag = "  <- rare, expect weaker accuracy" if count < 300 else ""
        print(f"    {emo:15s} {count:6d}  {bar}{flag}")


def main():
    print("Downloading GoEmotions (simplified, multi-label) from Hugging Face...")
    print("This needs an internet connection and may take a minute.\n")

    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    print("Converting splits...")
    train_df = convert_split(dataset["train"], "train")
    val_df   = convert_split(dataset["validation"], "validation")
    test_df  = convert_split(dataset["test"], "test")

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print(f"\nSaved:")
    print(f"  {OUTPUT_TRAIN}  ({len(train_df)} rows)")
    print(f"  {OUTPUT_VAL}    ({len(val_df)} rows)")
    print(f"  {OUTPUT_TEST}   ({len(test_df)} rows)")

    print_label_distribution(train_df)

    print("\nSample rows:")
    print(train_df[["text", "labels"]].head(5).to_string())

    print("\nDone. These three CSVs are ready to feed into the fine-tuning script.")
    print("Each row has a 'labels' column (comma-separated names) and one")
    print("binary 0/1 column per emotion — the fine-tuning script uses the")
    print("binary columns directly to build multi-label training targets.")


if __name__ == "__main__":
    main()
