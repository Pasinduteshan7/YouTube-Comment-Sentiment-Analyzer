"""
YouTube Comment Sentiment Analyser — FastAPI Backend
Version 2.0 (Refactored)

Routes:
  GET  /             → Health check
  GET  /last-analysis → Load previous analysis from CSV
  POST /analyse      → Analyse a single video
  POST /analyse-channel → Analyse multiple videos from a channel
  GET  /history      → Past analysis runs from MLflow
"""

__version__ = "2.0.0"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
from datetime import datetime
from collections import Counter
from contextlib import asynccontextmanager
from langdetect import detect, LangDetectException
import os
import time

from dotenv import load_dotenv
load_dotenv()

# ── local modules ──────────────────────────────────────────────────────
from schemas import AnalysisRequest, ChannelAnalysisRequest
from models import (
    load_models, predict_emotions_batch, predict_sentiment_batch, predict_toxicity_batch,
    get_sentiment_pipeline, EMOTION_LABELS, EMOTION_MODEL_PATH,
)
from analysis import (
    detect_mixed_sentiment, emotion_counts_from_lists, detect_topics,
    classify_emotional_fingerprint, find_conflicted_comments,
    compute_like_weighted_emotions, find_pin_suggestions, clean_comments,
    analyze_sentiment_over_time
)
from youtube import get_video_id, get_video_info, fetch_channel_videos
from suggestions import generate_suggestions
from fetcher import fetch_comments


# ── app setup ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="YouTube Sentiment Analyzer API", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("youtube-sentiment-analyser")


# ── simple in-memory cache ─────────────────────────────────────────────

_cache = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes


def get_cached(key: str):
    """Returns cached result if it exists and is not expired."""
    if key in _cache:
        result, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            print(f"Cache hit for {key}")
            return result
        else:
            del _cache[key]
    return None


def set_cache(key: str, value):
    """Stores a result in the cache."""
    _cache[key] = (value, time.time())


# ── helpers ────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def run_full_analysis(df: pd.DataFrame, video_info: dict, url: str) -> dict:
    """
    Core analysis pipeline shared by /analyse and /analyse-channel.
    Takes a cleaned DataFrame, runs all models, and returns the full result dict.
    """
    texts = df["text"].astype(str).tolist()
    total = len(df)

    # Language detection
    df["language"] = df["text"].apply(detect_language)

    # Sentiment analysis
    sent_results          = predict_sentiment_batch(texts)
    df["sentiment"]       = [r["label"] for r in sent_results]
    df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]

    # Multi-label emotion analysis
    emotion_lists  = predict_emotions_batch(texts)
    df["emotions"] = [",".join(emo_list) for emo_list in emotion_lists]
    df["emotion"]  = [emo_list[0] for emo_list in emotion_lists]

    # Toxicity analysis
    toxicity_results      = predict_toxicity_batch(texts)
    df["is_toxic"]        = [r["is_toxic"] for r in toxicity_results]
    df["toxicity_score"]  = [round(r["toxicity_score"], 3) for r in toxicity_results]

    # Mixed sentiment detection
    sent_pipe = get_sentiment_pipeline()
    mixed_data            = [detect_mixed_sentiment(t, sent_pipe) for t in texts]
    df["is_mixed"]        = [m["is_mixed"]                for m in mixed_data]
    df["part1_text"]      = [m.get("part1_text", "")      for m in mixed_data]
    df["part1_sentiment"] = [m.get("part1_sentiment", "") for m in mixed_data]
    df["part2_text"]      = [m.get("part2_text", "")      for m in mixed_data]
    df["part2_sentiment"] = [m.get("part2_sentiment", "") for m in mixed_data]

    # Aggregate metrics
    sentiment_counts = {
        "positive": int((df["sentiment"] == "positive").sum()),
        "neutral":  int((df["sentiment"] == "neutral").sum()),
        "negative": int((df["sentiment"] == "negative").sum()),
    }
    toxic_count = int(df["is_toxic"].sum())
    emotion_counts = emotion_counts_from_lists(emotion_lists)
    comments       = df.to_dict(orient="records")
    for i, c in enumerate(comments):
        c["emotions"] = emotion_lists[i]

    language_counts = df["language"].value_counts().head(10).to_dict()
    fingerprint     = classify_emotional_fingerprint(emotion_counts, total)
    conflicted      = find_conflicted_comments(comments)
    like_weighted   = compute_like_weighted_emotions(comments)
    topics          = detect_topics(comments)
    pin_suggestions = find_pin_suggestions(comments)
    suggestions_out = generate_suggestions(
        comments, sentiment_counts, emotion_counts,
        fingerprint, conflicted, like_weighted
    )

    timeline        = analyze_sentiment_over_time(df)

    return {
        "total":            total,
        "video_info":       video_info,
        "comments":         comments,
        "sentiment_counts": sentiment_counts,
        "emotion_counts":   emotion_counts,
        "suggestions":      suggestions_out,
        "topics":           topics,
        "fingerprint":      fingerprint,
        "conflicted":       conflicted,
        "like_weighted":    like_weighted,
        "toxic_count":      toxic_count,
        "language_counts":  language_counts,
        "pin_suggestions":  pin_suggestions,
        "timeline":         timeline,
        "df":               df,                  # for MLflow logging / CSV saving
        "emotion_lists":    emotion_lists,        # for channel aggregation
    }


# ── routes ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "version": __version__,
        "emotion_model": EMOTION_MODEL_PATH,
        "emotion_labels": len(EMOTION_LABELS),
    }


@app.get("/last-analysis")
def last_analysis():
    csv_path = "comments_analysed.csv"
    empty = {
        "comments": [], "video_info": {}, "sentiment_counts": {},
        "emotion_counts": {}, "suggestions": [], "topics": [],
        "total": 0, "fingerprint": {}, "conflicted": [], "like_weighted": {},
    }
    if not os.path.exists(csv_path):
        return empty
    df = pd.read_csv(csv_path)
    if df.empty:
        return empty

    # Ensure object dtype so NaN can be replaced by None
    df = df.astype(object).where(pd.notna(df), None)

    for col in ["is_mixed", "part1_text", "part1_sentiment", "part2_text", "part2_sentiment"]:
        if col not in df.columns:
            df[col] = False if col == "is_mixed" else ""
    if "emotions" not in df.columns:
        df["emotions"] = df.get("emotion", "neutral").apply(
            lambda x: [str(x)] if pd.notna(x) else ["neutral"]
        )
    else:
        df["emotions"] = df["emotions"].apply(
            lambda x: x.split(",") if isinstance(x, str) else ["neutral"]
        )

    if "language" not in df.columns:
        df["language"] = "unknown"
    language_counts = df["language"].value_counts().head(10).to_dict()

    comments = df.to_dict(orient="records")
    sentiment_counts = {
        "positive": int((df["sentiment"] == "positive").sum()),
        "neutral":  int((df["sentiment"] == "neutral").sum()),
        "negative": int((df["sentiment"] == "negative").sum()),
    }
    all_emotion_lists = [
        row["emotions"] if isinstance(row["emotions"], list) else [row["emotions"]]
        for row in comments
    ]
    e_counts       = emotion_counts_from_lists(all_emotion_lists)
    fingerprint    = classify_emotional_fingerprint(e_counts, len(df))
    conflicted     = find_conflicted_comments(comments)
    like_weighted  = compute_like_weighted_emotions(comments)
    pin_suggestions = find_pin_suggestions(comments)

    return {
        "total":            len(df),
        "video_info":       {},
        "comments":         comments,
        "sentiment_counts": sentiment_counts,
        "emotion_counts":   e_counts,
        "suggestions":      generate_suggestions(comments, sentiment_counts, e_counts,
                                                 fingerprint, conflicted, like_weighted),
        "topics":           detect_topics(comments),
        "fingerprint":      fingerprint,
        "conflicted":       conflicted,
        "like_weighted":    like_weighted,
        "toxic_count":      int(df["is_toxic"].sum()) if "is_toxic" in df.columns else 0,
        "language_counts":  language_counts,
        "pin_suggestions":  pin_suggestions,
    }


@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    try:
        video_id = get_video_id(request.url)

        # Check cache
        cache_key = f"{video_id}:{request.max_comments}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        df = fetch_comments(request.url, max_comments=request.max_comments)
        df = clean_comments(df)
        if df.empty:
            raise HTTPException(status_code=400, detail="No meaningful comments found after filtering.")

        video_info = get_video_info(video_id)
        result     = run_full_analysis(df, video_info, request.url)

        # Extract internal fields before returning
        analysis_df    = result.pop("df")
        emotion_lists  = result.pop("emotion_lists")
        total          = result["total"]
        sentiment_counts = result["sentiment_counts"]
        emotion_counts   = result["emotion_counts"]
        fingerprint      = result["fingerprint"]
        conflicted       = result["conflicted"]

        # Log to MLflow
        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("url",              request.url)
            mlflow.log_param("video_title",      video_info.get("title", "unknown"))
            mlflow.log_param("total_comments",   total)
            mlflow.log_param("emotion_model",    EMOTION_MODEL_PATH)
            mlflow.log_param("fingerprint",      fingerprint.get("profile", "unknown"))
            mlflow.log_metric("positive_pct",    round(sentiment_counts["positive"] / total * 100, 2))
            mlflow.log_metric("negative_pct",    round(sentiment_counts["negative"] / total * 100, 2))
            mlflow.log_metric("neutral_pct",     round(sentiment_counts["neutral"]  / total * 100, 2))
            mlflow.log_metric("avg_sentiment_score", round(analysis_df["sentiment_score"].mean(), 3))
            mlflow.log_metric("mixed_count",     int(analysis_df["is_mixed"].sum()))
            mlflow.log_metric("conflicted_count",len(conflicted))
            mlflow.log_metric("gratitude_pct",   round(emotion_counts.get("gratitude", 0) / total * 100, 2))
            mlflow.log_metric("realization_pct", round(emotion_counts.get("realization", 0) / total * 100, 2))
            for emo, cnt in emotion_counts.items():
                if cnt > 0:
                    mlflow.log_metric(f"emotion_{emo}", cnt)

        # Save CSV
        analysis_df.to_csv("comments_analysed.csv", index=False)

        # Cache the response
        response = {"status": "success", **result}
        set_cache(cache_key, response)

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyse-channel")
async def analyse_channel(request: ChannelAnalysisRequest):
    """
    Fetches the last N videos from a channel/playlist,
    runs full sentiment + emotion analysis on each,
    and returns per-video results plus cross-video comparison data.
    """
    try:
        videos = fetch_channel_videos(request.url, max_videos=request.max_videos)
        if not videos:
            raise HTTPException(status_code=400, detail="Could not find videos for this channel or playlist URL.")

        results = []
        all_channel_comments = []

        for video in videos:
            try:
                df = fetch_comments(video["url"], max_comments=request.comments_per_video)
                if df.empty:
                    continue

                df = clean_comments(df)
                if df.empty:
                    continue

                result = run_full_analysis(df, {}, video["url"])
                analysis_df   = result.pop("df")
                emotion_lists = result.pop("emotion_lists")

                total            = result["total"]
                sentiment_counts = result["sentiment_counts"]
                emotion_counts   = result["emotion_counts"]
                fingerprint      = result["fingerprint"]
                comments         = result["comments"]

                for c in comments:
                    c["video_title"] = video["title"]
                all_channel_comments.extend(comments)

                results.append({
                    "video_id":         video["video_id"],
                    "title":            video["title"],
                    "thumbnail":        video["thumbnail"],
                    "published":        video["published"],
                    "url":              video["url"],
                    "total":            total,
                    "sentiment_counts": sentiment_counts,
                    "emotion_counts":   emotion_counts,
                    "fingerprint":      fingerprint,
                    "positive_pct":     round(sentiment_counts["positive"] / max(total, 1) * 100, 1),
                    "negative_pct":     round(sentiment_counts["negative"] / max(total, 1) * 100, 1),
                    "top_emotions":     sorted(
                        [(e, c) for e, c in emotion_counts.items() if e not in ("approval", "neutral") and c > 0],
                        key=lambda x: -x[1]
                    )[:5],
                })
            except Exception as e:
                print(f"Error analysing {video['title']}: {e}")
                continue

        if not results:
            raise HTTPException(status_code=400, detail="Could not analyse any videos from this channel.")

        # Cross-video comparison metrics
        avg_positive = round(sum(r["positive_pct"] for r in results) / len(results), 1)
        avg_negative = round(sum(r["negative_pct"] for r in results) / len(results), 1)
        best_video   = max(results, key=lambda r: r["positive_pct"])
        worst_video  = max(results, key=lambda r: r["negative_pct"])
        fingerprints = [r["fingerprint"]["profile"] for r in results]
        most_common_profile = Counter(fingerprints).most_common(1)[0][0]

        return {
            "status":               "success",
            "total_videos":         len(results),
            "channel_url":          request.url,
            "avg_positive_pct":     avg_positive,
            "avg_negative_pct":     avg_negative,
            "best_received_video":  best_video["title"],
            "most_divisive_video":  worst_video["title"],
            "most_common_profile":  most_common_profile,
            "videos":               results,
            "latest_comments":      all_channel_comments[:50],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history():
    """Returns the last 15 analysis runs from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("youtube-sentiment-analyser")
        if not experiment:
            return {"runs": []}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=15,
        )
        history = []
        for run in runs:
            params  = run.data.params
            metrics = run.data.metrics
            history.append({
                "run_id":       run.info.run_id,
                "timestamp":    datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                "video_title":  params.get("video_title", "Unknown"),
                "url":          params.get("url", ""),
                "fingerprint":  params.get("fingerprint", ""),
                "positive_pct": metrics.get("positive_pct", 0),
                "negative_pct": metrics.get("negative_pct", 0),
                "total":        int(params.get("total_comments", 0)),
            })
        return {"runs": history}
    except Exception as e:
        return {"runs": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)