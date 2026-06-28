import pandas as pd
from transformers import pipeline

CONTRAST_WORDS = [
    " but ", " however ", " although ", " though ",
    " yet ", " despite ", " nevertheless ", " while ",
    " whereas ", " even though ", " on the other hand "
]

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

# ── load comments ──────────────────────────────────────────────
df    = pd.read_csv("comments.csv")
texts = df["text"].astype(str).tolist()
print(f"Loaded {len(texts)} comments. Running models...\n")

# ── load models ────────────────────────────────────────────────
print("Loading sentiment model...")
sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True, max_length=512
)

print("Loading emotion model...")
emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True, max_length=512
)

# ── run sentiment + emotion ────────────────────────────────────
print("Analysing comments (this takes 1-2 mins)...\n")
sent_results    = sentiment(texts, batch_size=16)
emotion_results = emotion(texts,   batch_size=16)

df["sentiment"]       = [r["label"] for r in sent_results]
df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]
df["emotion"]         = [r["label"] for r in emotion_results]
df["emotion_score"]   = [round(r["score"], 3) for r in emotion_results]

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
print("=== SENTIMENT BREAKDOWN ===")
print(df["sentiment"].value_counts().to_string())

print("\n=== EMOTION BREAKDOWN ===")
print(df["emotion"].value_counts().to_string())

mixed_count = df["is_mixed"].sum()
print(f"\n=== MIXED SENTIMENT ===")
print(f"Mixed comments detected: {mixed_count}")
if mixed_count > 0:
    print(df[df["is_mixed"] == True][["text", "part1_sentiment", "part2_sentiment"]].head(5).to_string())

print("\n=== SAMPLE RESULTS ===")
print(df[["text", "sentiment", "emotion", "is_mixed"]].head(10).to_string())
print("\nDone! Saved to comments_analysed.csv")