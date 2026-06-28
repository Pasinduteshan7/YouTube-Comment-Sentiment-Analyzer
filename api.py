from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import mlflow
import pandas as pd
from datetime import datetime
from fetcher import fetch_comments
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="YouTube Sentiment Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("youtube-sentiment-analyser")

class AnalysisRequest(BaseModel):
    url: str
    max_comments: int = 100

sentiment_model = None
emotion_model   = None

def load_models():
    global sentiment_model, emotion_model
    print("Loading models...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True, max_length=512
    )
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        truncation=True, max_length=512
    )
    print("✓ Models loaded!")

@app.on_event("startup")
async def startup_event():
    load_models()

# ── helpers ────────────────────────────────────────────────────

def get_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    return url.split("/")[-1].split("?")[0]

def get_video_info(video_id: str) -> dict:
    try:
        api_key = os.getenv("YOUTUBE_API_KEY")
        youtube = build("youtube", "v3", developerKey=api_key)
        res = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        if not res.get("items"):
            return {}
        item    = res["items"][0]
        snippet = item["snippet"]
        stats   = item.get("statistics", {})
        return {
            "title":         snippet.get("title", ""),
            "channel":       snippet.get("channelTitle", ""),
            "thumbnail":     snippet["thumbnails"]["high"]["url"],
            "published":     snippet.get("publishedAt", "")[:10],
            "view_count":    int(stats.get("viewCount", 0)),
            "like_count":    int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        }
    except Exception as e:
        print(f"Video info error: {e}")
        return {}

# ── mixed sentiment ────────────────────────────────────────────

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

# ── topic modelling ────────────────────────────────────────────

def detect_topics(comments: list) -> list:
    topic_rules = {
        "Audio / sound quality":    ["audio", "sound", "mic", "microphone", "volume", "hear", "noise", "echo", "bass", "loud", "quiet"],
        "Video / visual quality":   ["video quality", "resolution", "blurry", "4k", "hd", "1080", "720", "camera", "lighting", "dark", "bright"],
        "Content length / pacing":  ["long", "short", "slow", "fast", "boring", "skip", "too long", "too short", "pacing", "duration", "minute"],
        "Clarity / explanation":    ["confus", "unclear", "hard to follow", "explain", "understand", "lost", "complex", "simple", "clear", "example"],
        "Request for more content": ["part 2", "next video", "more", "series", "continue", "follow up", "sequel", "episode", "please make"],
        "Positive praise":          ["amazing", "great", "love", "best", "awesome", "fantastic", "perfect", "excellent", "helpful", "thank"],
        "Criticism / complaint":    ["bad", "worst", "hate", "terrible", "awful", "waste", "dislike", "disappoint", "wrong", "mistake"],
        "Question / curiosity":     ["?", "how", "why", "what", "when", "where", "can you", "could you", "does", "is it"],
        "Humour / meme":            ["lol", "lmao", "haha", "funny", "joke", "meme", "bruh", "bro", "literally"],
    }
    topic_counts   = {t: 0 for t in topic_rules}
    topic_examples = {t: [] for t in topic_rules}
    for c in comments:
        text_lower = c["text"].lower()
        for topic, keywords in topic_rules.items():
            if any(kw in text_lower for kw in keywords):
                topic_counts[topic] += 1
                if len(topic_examples[topic]) < 2:
                    topic_examples[topic].append(c["text"][:80])
    results = [
        {
            "topic":    topic,
            "count":    count,
            "percent":  round(count / max(len(comments), 1) * 100, 1),
            "examples": topic_examples[topic],
        }
        for topic, count in topic_counts.items() if count > 0
    ]
    return sorted(results, key=lambda x: x["count"], reverse=True)

# ── suggestions ────────────────────────────────────────────────

def generate_suggestions(comments: list, sentiment_counts: dict, emotion_counts: dict) -> list:
    negative = [c["text"] for c in comments if c["sentiment"] == "negative"][:20]
    positive = [c["text"] for c in comments if c["sentiment"] == "positive"][:10]
    total    = max(len(comments), 1)
    neg_pct  = round(sentiment_counts["negative"] / total * 100)
    pos_pct  = round(sentiment_counts["positive"] / total * 100)
    top_emo  = max(emotion_counts, key=emotion_counts.get)
    neg_text = " ".join(negative).lower()
    pos_text = " ".join(positive).lower()

    suggestions = []

    if neg_pct > 30:
        suggestions.append({"type": "warning", "title": "High negative sentiment detected",
            "detail": f"{neg_pct}% of comments are negative. Review top negative comments and address concerns in your next video."})
    if neg_pct <= 15:
        suggestions.append({"type": "success", "title": "Excellent audience reception",
            "detail": f"Only {neg_pct}% negative comments. Your audience is highly satisfied — keep this content style."})
    if pos_pct >= 40:
        suggestions.append({"type": "success", "title": "Strong positive engagement",
            "detail": f"{pos_pct}% positive comments. Consider making more content in this topic area."})
    if top_emo == "anger":
        suggestions.append({"type": "warning", "title": "Anger is the dominant emotion",
            "detail": "Many viewers expressed anger. Check if content was controversial or if the title/thumbnail set wrong expectations."})
    if top_emo == "joy":
        suggestions.append({"type": "success", "title": "Joy is the dominant emotion",
            "detail": "Viewers are genuinely happy with this content. Replicate this style."})
    if top_emo == "surprise":
        suggestions.append({"type": "info", "title": "Viewers were surprised",
            "detail": "Surprise is the top emotion — your content subverted expectations. Use this for higher engagement."})
    if top_emo == "sadness":
        suggestions.append({"type": "info", "title": "Sadness is the dominant emotion",
            "detail": "Viewers felt emotionally moved. Consider leaning into emotional storytelling."})
    if any(w in neg_text for w in ["long", "slow", "boring", "skip", "too long"]):
        suggestions.append({"type": "warning", "title": "Pacing feedback detected",
            "detail": "Negative comments mention length or pacing. Consider tighter editing and chapter markers."})
    if any(w in neg_text for w in ["audio", "sound", "hear", "mic", "volume"]):
        suggestions.append({"type": "warning", "title": "Audio quality complaints",
            "detail": "Viewers are flagging audio issues. Better microphone or post-processing could significantly help."})
    if any(w in neg_text for w in ["explain", "confus", "understand", "unclear", "hard to follow"]):
        suggestions.append({"type": "warning", "title": "Clarity issues reported",
            "detail": "Viewers find the content hard to follow. Add examples, summaries, or chapter markers."})
    if any(w in neg_text for w in ["clickbait", "mislead", "thumbnail", "title"]):
        suggestions.append({"type": "warning", "title": "Thumbnail / title mismatch",
            "detail": "Viewers feel the thumbnail or title was misleading. Make sure content delivers on the promise."})
    if any(w in pos_text for w in ["part 2", "more", "next", "series", "continue", "follow up"]):
        suggestions.append({"type": "info", "title": "Viewers want more content",
            "detail": "Positive comments ask for follow-up. A part 2 or series would perform well."})
    if not suggestions:
        suggestions.append({"type": "info", "title": "Neutral audience response",
            "detail": "Comments are mostly neutral. Try ending videos with a direct question to boost engagement."})
    return suggestions

# ── routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/last-analysis")
def last_analysis():
    csv_path = "comments_analysed.csv"
    if not os.path.exists(csv_path):
        return {"comments": [], "video_info": {}, "sentiment_counts": {}, "emotion_counts": {}, "suggestions": [], "topics": [], "total": 0}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"comments": [], "video_info": {}, "sentiment_counts": {}, "emotion_counts": {}, "suggestions": [], "topics": [], "total": 0}

    for col in ["is_mixed", "part1_text", "part1_sentiment", "part2_text", "part2_sentiment"]:
        if col not in df.columns:
            df[col] = False if col == "is_mixed" else ""

    comments = df.to_dict(orient="records")
    sentiment_counts = {
        "positive": int((df["sentiment"] == "positive").sum()),
        "neutral":  int((df["sentiment"] == "neutral").sum()),
        "negative": int((df["sentiment"] == "negative").sum()),
    }
    emotion_counts = {e: int((df["emotion"] == e).sum()) for e in ["joy","neutral","surprise","anger","sadness","fear","disgust"]}
    return {
        "total": len(df), "video_info": {}, "comments": comments,
        "sentiment_counts": sentiment_counts, "emotion_counts": emotion_counts,
        "suggestions": generate_suggestions(comments, sentiment_counts, emotion_counts),
        "topics": detect_topics(comments),
    }

@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    try:
        df = fetch_comments(request.url, max_comments=request.max_comments)
        if df.empty:
            raise HTTPException(status_code=400, detail="No comments found for this video.")

        video_id   = get_video_id(request.url)
        video_info = get_video_info(video_id)

        texts           = df["text"].astype(str).tolist()
        sent_results    = sentiment_model(texts, batch_size=16)
        emotion_results = emotion_model(texts, batch_size=16)

        df["sentiment"]       = [r["label"] for r in sent_results]
        df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]
        df["emotion"]         = [r["label"] for r in emotion_results]
        df["emotion_score"]   = [round(r["score"], 3) for r in emotion_results]

        # mixed sentiment detection
        mixed_data = [detect_mixed_sentiment(t, sentiment_model) for t in texts]
        df["is_mixed"]        = [m["is_mixed"]                for m in mixed_data]
        df["part1_text"]      = [m.get("part1_text", "")      for m in mixed_data]
        df["part1_sentiment"] = [m.get("part1_sentiment", "") for m in mixed_data]
        df["part2_text"]      = [m.get("part2_text", "")      for m in mixed_data]
        df["part2_sentiment"] = [m.get("part2_sentiment", "") for m in mixed_data]

        total = len(df)
        sentiment_counts = {
            "positive": int((df["sentiment"] == "positive").sum()),
            "neutral":  int((df["sentiment"] == "neutral").sum()),
            "negative": int((df["sentiment"] == "negative").sum()),
        }
        emotion_counts = {e: int((df["emotion"] == e).sum()) for e in ["joy","neutral","surprise","anger","sadness","fear","disgust"]}
        comments    = df.to_dict(orient="records")
        suggestions = generate_suggestions(comments, sentiment_counts, emotion_counts)
        topics      = detect_topics(comments)

        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("url",            request.url)
            mlflow.log_param("video_title",    video_info.get("title", "unknown"))
            mlflow.log_param("total_comments", total)
            mlflow.log_metric("positive_pct",  round(sentiment_counts["positive"] / total * 100, 2))
            mlflow.log_metric("negative_pct",  round(sentiment_counts["negative"] / total * 100, 2))
            mlflow.log_metric("neutral_pct",   round(sentiment_counts["neutral"]  / total * 100, 2))
            mlflow.log_metric("avg_sentiment_score", round(df["sentiment_score"].mean(), 3))
            mlflow.log_metric("mixed_count",   int(df["is_mixed"].sum()))
            for emo, cnt in emotion_counts.items():
                mlflow.log_metric(f"emotion_{emo}", cnt)

        df.to_csv("comments_analysed.csv", index=False)

        return {
            "status": "success", "total": total,
            "video_info": video_info, "comments": comments,
            "sentiment_counts": sentiment_counts, "emotion_counts": emotion_counts,
            "suggestions": suggestions, "topics": topics,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)