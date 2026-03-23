import pandas as pd
from transformers import pipeline

# ── load your comments ──
df = pd.read_csv("comments.csv")
texts = df["text"].astype(str).tolist()

print(f"Loaded {len(texts)} comments. Running models...\n")

# ── 1. Sentiment model ──
print("Loading sentiment model...")
sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True, max_length=512
)

# ── 2. Emotion model ──
print("Loading emotion model...")
emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True, max_length=512
)

# ── run both models on every comment ──
print("Analysing comments (this takes 1-2 mins)...\n")

sent_results   = sentiment(texts, batch_size=16)
emotion_results = emotion(texts, batch_size=16)

# ── add results back to dataframe ──
df["sentiment"]       = [r["label"] for r in sent_results]
df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]
df["emotion"]         = [r["label"] for r in emotion_results]
df["emotion_score"]   = [round(r["score"], 3) for r in emotion_results]

# ── save enriched CSV ──
df.to_csv("comments_analysed.csv", index=False)

# ── quick summary ──
print("=== SENTIMENT BREAKDOWN ===")
print(df["sentiment"].value_counts().to_string())
print("\n=== EMOTION BREAKDOWN ===")
print(df["emotion"].value_counts().to_string())
print("\n=== SAMPLE RESULTS ===")
print(df[["text","sentiment","emotion"]].head(10).to_string())
print("\nDone! Saved to comments_analysed.csv")