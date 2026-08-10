"""
Standalone sentiment + emotion analyser for YouTube comments.
Uses the same fine-tuned 28-label multi-label emotion model as api.py.

Usage:
    python analyser.py
"""

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

CONTRAST_WORDS = [
    " but ", " however ", " although ", " though ",
    " yet ", " despite ", " nevertheless ", " while ",
    " whereas ", " even though ", " on the other hand "
]

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]

EMOTION_MODEL_PATH = "./fine-tuned-emotion-model-multilingual"
EMOTION_THRESHOLD  = 0.3


def detect_mixed_sentiment(text: str, sent_pipeline) -> dict:
    text_lower = text.lower()
    split_at   = None
    for cw in CONTRAST_WORDS:
        idx = text_lower.find(cw)
        if idx != -1:
            split_at = idx + len(cw) - 1
            break
    if split_at is None:
        return {"is_mixed": False}
    part1 = text[:split_at].strip()
    part2 = text[split_at:].strip()
    if len(part1.split()) < 3 or len(part2.split()) < 3:
        return {"is_mixed": False}
    r1 = sent_pipeline(part1, truncation=True, max_length=512)[0]
    r2 = sent_pipeline(part2, truncation=True, max_length=512)[0]
    if r1["label"] == r2["label"]:
        return {"is_mixed": False}
    return {
        "is_mixed":        True,
        "part1_text":      part1,
        "part1_sentiment": r1["label"],
        "part1_score":     round(r1["score"], 3),
        "part2_text":      part2,
        "part2_sentiment": r2["label"],
        "part2_score":     round(r2["score"], 3),
    }


def predict_emotions_batch(texts: list, tokenizer, model, device, batch_size=32) -> list:
    """Multi-label emotion prediction using sigmoid thresholding (consistent with api.py)."""
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i: i + batch_size]
        encoded = tokenizer(
            batch, truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        for row in probs:
            detected = [
                EMOTION_LABELS[j]
                for j, score in enumerate(row)
                if score >= EMOTION_THRESHOLD
            ]
            if not detected:
                detected = [EMOTION_LABELS[int(row.argmax())]]
            all_results.append(detected)
    return all_results


if __name__ == "__main__":
    import os

    # ── load comments ──────────────────────────────────────────────
    csv_path = "comments.csv"
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found. Run fetcher.py first.")
        exit(1)

    df    = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    print(f"Loaded {len(texts)} comments. Running models...\n")

    # ── load models ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading sentiment model...")
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True, max_length=512
    )

    print("Loading fine-tuned 28-label emotion model...")
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
    emotion_model     = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_PATH
    ).to(device)
    emotion_model.eval()

    # ── run sentiment ──────────────────────────────────────────────
    print("Analysing sentiment (this takes 1-2 mins)...\n")
    sent_results = sentiment(texts, batch_size=16)

    df["sentiment"]       = [r["label"] for r in sent_results]
    df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]

    # ── run multi-label emotions ───────────────────────────────────
    print("Analysing emotions (28-label multi-label)...\n")
    emotion_lists  = predict_emotions_batch(texts, emotion_tokenizer, emotion_model, device)
    df["emotions"] = [",".join(emo_list) for emo_list in emotion_lists]
    df["emotion"]  = [emo_list[0] for emo_list in emotion_lists]

    # ── run mixed sentiment detection ──────────────────────────────
    print("Detecting mixed sentiment...")
    mixed_data = [detect_mixed_sentiment(t, sentiment) for t in texts]

    df["is_mixed"]        = [m["is_mixed"]                for m in mixed_data]
    df["part1_text"]      = [m.get("part1_text", "")      for m in mixed_data]
    df["part1_sentiment"] = [m.get("part1_sentiment", "") for m in mixed_data]
    df["part2_text"]      = [m.get("part2_text", "")      for m in mixed_data]
    df["part2_sentiment"] = [m.get("part2_sentiment", "") for m in mixed_data]

    # ── save ───────────────────────────────────────────────────────
    df.to_csv("comments_analysed.csv", index=False)

    # ── summary ────────────────────────────────────────────────────
    print("\n=== SENTIMENT BREAKDOWN ===")
    print(df["sentiment"].value_counts().to_string())

    print("\n=== EMOTION BREAKDOWN (top 10) ===")
    from collections import Counter
    all_emotions = []
    for emo_list in emotion_lists:
        all_emotions.extend(emo_list)
    emotion_counter = Counter(all_emotions)
    for emo, count in emotion_counter.most_common(10):
        print(f"  {emo:15s} {count}")

    mixed_count = df["is_mixed"].sum()
    print(f"\n=== MIXED SENTIMENT ===")
    print(f"Mixed comments detected: {mixed_count}")
    if mixed_count > 0:
        print(df[df["is_mixed"] == True][["text", "part1_sentiment", "part2_sentiment"]].head(5).to_string())

    print("\n=== SAMPLE RESULTS ===")
    print(df[["text", "sentiment", "emotions", "is_mixed"]].head(10).to_string())
    print("\nDone! Saved to comments_analysed.csv")