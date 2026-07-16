from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mlflow
import pandas as pd
from datetime import datetime
from fetcher import fetch_comments
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import re
from contextlib import asynccontextmanager
from langdetect import detect, LangDetectException
import groq

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="YouTube Sentiment Analyzer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("youtube-sentiment-analyser")

SENTIMENT_MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_PATH   = "./fine-tuned-emotion-model-multilingual"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]

EMOTION_THRESHOLD = 0.3

class AnalysisRequest(BaseModel):
    url: str
    max_comments: int = 100

class ChannelAnalysisRequest(BaseModel):
    url: str
    max_videos: int = 5
    comments_per_video: int = 100

sentiment_pipeline = None
emotion_tokenizer  = None
emotion_model      = None
device             = None

def load_models():
    global sentiment_pipeline, emotion_tokenizer, emotion_model, device
    from transformers import pipeline

    device_id = 0 if torch.cuda.is_available() else -1
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading sentiment model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_PATH,
        truncation=True, max_length=512,
        device=device_id,
    )

    print("Loading fine-tuned emotion model...")
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
    emotion_model     = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_PATH
    ).to(device)
    emotion_model.eval()
    print("Models loaded!")


# ── emotion inference ──────────────────────────────────────────────────

def predict_emotions_batch(texts: list, batch_size: int = 32) -> list:
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i: i + batch_size]
        encoded = emotion_tokenizer(
            batch, truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = emotion_model(**encoded).logits
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


def emotion_counts_from_lists(emotion_lists: list) -> dict:
    counts = {e: 0 for e in EMOTION_LABELS}
    for emo_list in emotion_lists:
        for emo in emo_list:
            if emo in counts:
                counts[emo] += 1
    return counts


# ── helpers ────────────────────────────────────────────────────────────

def fetch_channel_videos(channel_url: str, max_videos: int = 10) -> list:
    """
    Given a channel URL or playlist URL, returns a list of dicts:
    [{"video_id": "...", "title": "...", "thumbnail": "...", "published": "..."}]
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)

    # resolve channel URL to uploads playlist ID
    if "@" in channel_url or "channel/" in channel_url:
        # extract channel handle or ID
        if "@" in channel_url:
            handle = channel_url.split("@")[-1].split("/")[0].split("?")[0]
            res = youtube.search().list(
                part="snippet", q=handle, type="channel", maxResults=1
            ).execute()
            if not res.get("items"):
                return []
            channel_id = res["items"][0]["snippet"]["channelId"]
        else:
            channel_id = channel_url.split("channel/")[-1].split("/")[0].split("?")[0]

        channel_res = youtube.channels().list(
            part="contentDetails", id=channel_id
        ).execute()
        if not channel_res.get("items"):
            return []
        uploads_playlist = channel_res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    elif "playlist?list=" in channel_url:
        uploads_playlist = channel_url.split("list=")[-1].split("&")[0]
    else:
        return []

    # fetch videos from playlist
    videos = []
    next_page = None
    while len(videos) < max_videos:
        res = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist,
            maxResults=min(50, max_videos - len(videos)),
            pageToken=next_page,
        ).execute()
        for item in res.get("items", []):
            snippet = item["snippet"]
            video_id = snippet["resourceId"]["videoId"]
            videos.append({
                "video_id":  video_id,
                "title":     snippet.get("title", ""),
                "thumbnail": snippet["thumbnails"].get("high", {}).get("url", ""),
                "published": snippet.get("publishedAt", "")[:10],
                "url":       f"https://www.youtube.com/watch?v={video_id}",
            })
        next_page = res.get("nextPageToken")
        if not next_page:
            break

    return videos[:max_videos]


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


# ── mixed sentiment ────────────────────────────────────────────────────

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


# ── topic modelling ────────────────────────────────────────────────────

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
        text_lower = str(c.get("text", "")).lower()
        for topic, keywords in topic_rules.items():
            if any(kw in text_lower for kw in keywords):
                topic_counts[topic] += 1
                if len(topic_examples[topic]) < 2:
                    topic_examples[topic].append(str(c.get("text", ""))[:80])
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


# ── emotional fingerprint ──────────────────────────────────────────────

def classify_emotional_fingerprint(emotion_counts: dict, total: int) -> dict:
    if total == 0:
        return {"profile": "Unknown", "description": "Not enough data."}

    def pct(emo):
        return round(emotion_counts.get(emo, 0) / total * 100, 1)

    gratitude_pct      = pct("gratitude")
    love_pct           = pct("love")
    admiration_pct     = pct("admiration")
    joy_pct            = pct("joy")
    curiosity_pct      = pct("curiosity")
    confusion_pct      = pct("confusion")
    realization_pct    = pct("realization")
    amusement_pct      = pct("amusement")
    excitement_pct     = pct("excitement")
    surprise_pct       = pct("surprise")
    anger_pct          = pct("anger")
    annoyance_pct      = pct("annoyance")
    disappointment_pct = pct("disappointment")
    sadness_pct        = pct("sadness")
    optimism_pct       = pct("optimism")
    caring_pct         = pct("caring")

    if gratitude_pct >= 8 and (love_pct + admiration_pct) >= 20:
        return {
            "profile": "Creator Loyalty",
            "description": (
                f"Viewers feel personally connected and grateful ({gratitude_pct}% gratitude, "
                f"{love_pct}% love). This content builds long-term subscriber loyalty."
            ),
        }
    if realization_pct >= 10 and (curiosity_pct + confusion_pct) >= 10:
        return {
            "profile": "Mind-Opening",
            "description": (
                f"This video genuinely shifted viewer perspectives ({realization_pct}% realization). "
                f"Educational or opinion content that changes thinking has very high share rates."
            ),
        }
    if amusement_pct >= 12 and (surprise_pct + excitement_pct) >= 10:
        return {
            "profile": "Entertainment Hit",
            "description": (
                f"High amusement ({amusement_pct}%) and excitement/surprise. "
                f"Viewers were genuinely entertained and surprised. Strong viral potential."
            ),
        }
    if admiration_pct >= 25 and joy_pct >= 15:
        return {
            "profile": "Skill Showcase",
            "description": (
                f"Admiration ({admiration_pct}%) is the dominant emotion alongside joy ({joy_pct}%). "
                f"Viewers are impressed by demonstrated skill or craft. Great for personal brand building."
            ),
        }
    if (curiosity_pct + confusion_pct) >= 20 and realization_pct >= 5:
        return {
            "profile": "Tutorial / Explainer",
            "description": (
                f"High curiosity ({curiosity_pct}%) and confusion ({confusion_pct}%) indicate viewers "
                f"came with questions. Consider adding clearer chapter markers or summaries."
            ),
        }
    if (anger_pct + annoyance_pct + disappointment_pct) >= 15:
        return {
            "profile": "Controversial / Divisive",
            "description": (
                f"Negative emotions are elevated: anger ({anger_pct}%), annoyance ({annoyance_pct}%), "
                f"disappointment ({disappointment_pct}%). Review whether expectations were set correctly."
            ),
        }
    if sadness_pct >= 10 and (love_pct + caring_pct) >= 15:
        return {
            "profile": "Emotional Storytelling",
            "description": (
                f"Sadness ({sadness_pct}%) alongside love and caring. Viewers felt emotionally moved. "
                f"This content resonates deeply. Consider more in this style."
            ),
        }
    if optimism_pct >= 10 and (excitement_pct + joy_pct) >= 15:
        return {
            "profile": "Motivational",
            "description": (
                f"Optimism ({optimism_pct}%) and excitement/joy dominate. "
                f"Viewers feel uplifted and energised. Strong potential for repeat viewing."
            ),
        }
    if excitement_pct >= 15 and surprise_pct >= 8:
        return {
            "profile": "High Energy",
            "description": (
                f"Excitement ({excitement_pct}%) and surprise ({surprise_pct}%). "
                f"Viewers are energised and caught off guard in a good way."
            ),
        }

    top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    top_3 = [e for e, c in top_emotions if e not in ("neutral", "approval") and c > 0][:3]
    return {
        "profile": "Mixed Reception",
        "description": (
            f"No single emotional theme dominates. Top emotions: {', '.join(top_3)}. "
            f"Try a stronger hook or clearer emotional direction in future videos."
        ),
    }


# ── emotion conflict detection ─────────────────────────────────────────

def find_conflicted_comments(comments: list) -> list:
    conflict_pairs = [
        ("admiration",  "disappointment"),
        ("joy",         "sadness"),
        ("excitement",  "fear"),
        ("love",        "anger"),
        ("optimism",    "disappointment"),
        ("admiration",  "anger"),
        ("joy",         "remorse"),
        ("approval",    "disapproval"),
        ("excitement",  "annoyance"),
        ("gratitude",   "disappointment"),
    ]
    conflicted = []
    for c in comments:
        emotions = c.get("emotions", [])
        if isinstance(emotions, str):
            emotions = [e.strip() for e in emotions.split(",") if e.strip()]
        found_pairs = []
        for e1, e2 in conflict_pairs:
            if e1 in emotions and e2 in emotions:
                found_pairs.append(f"{e1} + {e2}")
        if found_pairs:
            conflicted.append({
                "text":          str(c.get("text", ""))[:120],
                "emotions":      emotions,
                "conflict_pair": found_pairs[0],
                "sentiment":     c.get("sentiment", ""),
                "likes":         c.get("likes", 0),
            })
    conflicted.sort(key=lambda x: x["likes"], reverse=True)
    return conflicted[:8]


# ── like-weighted emotion scoring ──────────────────────────────────────

def compute_like_weighted_emotions(comments: list) -> dict:
    weighted = {e: 0 for e in EMOTION_LABELS}
    for c in comments:
        likes    = max(int(c.get("likes", 0)), 1)
        emotions = c.get("emotions", [])
        if isinstance(emotions, str):
            emotions = [e.strip() for e in emotions.split(",") if e.strip()]
        for emo in emotions:
            if emo in weighted:
                weighted[emo] += likes
    filtered = {k: v for k, v in weighted.items() if k not in ("approval", "neutral") and v > 0}
    top5 = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:5]
    return {emo: score for emo, score in top5}


# ── suggestions ────────────────────────────────────────────────────────

def generate_suggestions(
    comments: list,
    sentiment_counts: dict,
    emotion_counts: dict,
    fingerprint: dict = None,
    conflicted: list = None,
    like_weighted: dict = None,
):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return [{"type": "warning", "title": "API Key Missing", "detail": "Add GROQ_API_KEY to your .env to enable the AI creator brief."}]
        
    try:
        client = groq.Groq(api_key=api_key)
        
        total = max(len(comments), 1)
        top_comments = sorted(comments, key=lambda x: int(x.get("likes", 0)), reverse=True)[:40]
        comment_texts = "\n".join([f"- (Likes: {c.get('likes', 0)}) {c.get('text', '')}" for c in top_comments])
        
        prompt = f"""You are an expert YouTube Strategist. Your client has just uploaded a video (or you are analysing their channel).
Based on the following data, write a 3-4 paragraph strategic brief for the creator.
Focus on specific insights, viewer demands, and actionable advice. DO NOT use generic advice. 
Reference specific viewer comments if relevant (e.g., "Several viewers asked for a Luke Cage crossover").
Do not include a greeting or sign-off, just output the brief in Markdown format.

DATA:
Total Comments Analysed: {total}
Sentiment: {sentiment_counts}
Emotions: {emotion_counts}
Emotional Fingerprint: {fingerprint.get('profile', 'Unknown') if fingerprint else 'Unknown'}

Top Comments:
{comment_texts}
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert YouTube Strategist. Output markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq API Error: {e}")
        return [{"type": "warning", "title": "AI Brief Failed", "detail": f"Could not generate report: {str(e)}"}]


# ── helpers (clean, lang, pin) ─────────────────────────────────────────

def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes low-quality comments before inference:
    - Pure emoji or symbol-only comments
    - Comments under 5 meaningful characters
    - Exact duplicates
    """
    def is_meaningful(text: str) -> bool:
        text = str(text).strip()
        # remove emoji and symbols, check what remains
        cleaned = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
        cleaned = cleaned.strip()
        return len(cleaned) >= 5

    original_count = len(df)
    df = df[df["text"].apply(is_meaningful)]
    df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)
    removed = original_count - len(df)
    if removed > 0:
        print(f"Filtered {removed} low-quality comments ({original_count} -> {len(df)})")
    return df

def detect_language(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"

def find_pin_suggestions(comments: list) -> dict:
    """
    Identifies the 3 comments most worth a creator reply:
    1. Best question — highest likes among comments with curiosity emotion
    2. Best conflicted — highest likes among comments with both approval and disapproval
    3. Best criticism — highest likes among negative sentiment comments
    """
    def get_emotions(c):
        e = c.get("emotions", [])
        if isinstance(e, str):
            return [x.strip() for x in e.split(",") if x.strip()]
        return e if isinstance(e, list) else []

    # 1. Best question
    questions = [c for c in comments if "curiosity" in get_emotions(c)]
    best_question = max(questions, key=lambda c: int(c.get("likes", 0)), default=None)

    # 2. Best conflicted (approval + disapproval)
    conflicted = [
        c for c in comments
        if "approval" in get_emotions(c) and "disapproval" in get_emotions(c)
    ]
    best_conflicted = max(conflicted, key=lambda c: int(c.get("likes", 0)), default=None)

    # 3. Best criticism (negative sentiment, most liked)
    criticisms = [c for c in comments if c.get("sentiment") == "negative"]
    best_criticism = max(criticisms, key=lambda c: int(c.get("likes", 0)), default=None)

    def fmt(c):
        if not c:
            return None
        return {
            "text":      str(c.get("text", ""))[:150],
            "likes":     c.get("likes", 0),
            "emotions":  get_emotions(c),
            "sentiment": c.get("sentiment", ""),
        }

    return {
        "best_question":   fmt(best_question),
        "best_conflicted": fmt(best_conflicted),
        "best_criticism":  fmt(best_criticism),
    }

# ── routes ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "emotion_model": EMOTION_MODEL_PATH}


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
        
    # Replace NaN with None for JSON compliance
    df = df.where(pd.notna(df), None)

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
    emotion_counts = emotion_counts_from_lists(all_emotion_lists)
    fingerprint    = classify_emotional_fingerprint(emotion_counts, len(df))
    conflicted     = find_conflicted_comments(comments)
    like_weighted  = compute_like_weighted_emotions(comments)
    pin_suggestions = find_pin_suggestions(comments)

    return {
        "total":          len(df),
        "video_info":     {},
        "comments":       comments,
        "sentiment_counts": sentiment_counts,
        "emotion_counts": emotion_counts,
        "suggestions":    generate_suggestions(comments, sentiment_counts, emotion_counts,
                                               fingerprint, conflicted, like_weighted),
        "topics":         detect_topics(comments),
        "fingerprint":    fingerprint,
        "conflicted":     conflicted,
        "like_weighted":  like_weighted,
        "language_counts": language_counts,
        "pin_suggestions": pin_suggestions,
    }


@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    try:
        df = fetch_comments(request.url, max_comments=request.max_comments)
        df = clean_comments(df)
        if df.empty:
            raise HTTPException(status_code=400, detail="No meaningful comments found after filtering.")

        video_id   = get_video_id(request.url)
        video_info = get_video_info(video_id)
        texts      = df["text"].astype(str).tolist()

        df["language"] = df["text"].apply(detect_language)

        sent_results          = sentiment_pipeline(texts, batch_size=16)
        df["sentiment"]       = [r["label"] for r in sent_results]
        df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]

        emotion_lists  = predict_emotions_batch(texts)
        df["emotions"] = [",".join(emo_list) for emo_list in emotion_lists]
        df["emotion"]  = [emo_list[0] for emo_list in emotion_lists]

        mixed_data            = [detect_mixed_sentiment(t, sentiment_pipeline) for t in texts]
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
        emotion_counts = emotion_counts_from_lists(emotion_lists)
        comments       = df.to_dict(orient="records")
        for i, c in enumerate(comments):
            c["emotions"] = emotion_lists[i]

        language_counts = df["language"].value_counts().head(10).to_dict()
        fingerprint   = classify_emotional_fingerprint(emotion_counts, total)
        conflicted    = find_conflicted_comments(comments)
        like_weighted = compute_like_weighted_emotions(comments)
        suggestions   = generate_suggestions(comments, sentiment_counts, emotion_counts,
                                             fingerprint, conflicted, like_weighted)
        topics        = detect_topics(comments)
        pin_suggestions = find_pin_suggestions(comments)

        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("url",              request.url)
            mlflow.log_param("video_title",      video_info.get("title", "unknown"))
            mlflow.log_param("total_comments",   total)
            mlflow.log_param("emotion_model",    EMOTION_MODEL_PATH)
            mlflow.log_param("fingerprint",      fingerprint.get("profile", "unknown"))
            mlflow.log_metric("positive_pct",    round(sentiment_counts["positive"] / total * 100, 2))
            mlflow.log_metric("negative_pct",    round(sentiment_counts["negative"] / total * 100, 2))
            mlflow.log_metric("neutral_pct",     round(sentiment_counts["neutral"]  / total * 100, 2))
            mlflow.log_metric("avg_sentiment_score", round(df["sentiment_score"].mean(), 3))
            mlflow.log_metric("mixed_count",     int(df["is_mixed"].sum()))
            mlflow.log_metric("conflicted_count",len(conflicted))
            mlflow.log_metric("gratitude_pct",   round(emotion_counts.get("gratitude", 0) / total * 100, 2))
            mlflow.log_metric("realization_pct", round(emotion_counts.get("realization", 0) / total * 100, 2))
            for emo, cnt in emotion_counts.items():
                if cnt > 0:
                    mlflow.log_metric(f"emotion_{emo}", cnt)

        df.to_csv("comments_analysed.csv", index=False)

        return {
            "status":           "success",
            "total":            total,
            "video_info":       video_info,
            "comments":         comments,
            "sentiment_counts": sentiment_counts,
            "emotion_counts":   emotion_counts,
            "suggestions":      suggestions,
            "topics":           topics,
            "fingerprint":      fingerprint,
            "conflicted":       conflicted,
            "like_weighted":    like_weighted,
            "language_counts":  language_counts,
            "pin_suggestions":  pin_suggestions,
        }
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

                texts = df["text"].astype(str).tolist()

                sent_results    = sentiment_pipeline(texts, batch_size=16)
                df["sentiment"] = [r["label"] for r in sent_results]
                df["sentiment_score"] = [round(r["score"], 3) for r in sent_results]

                emotion_lists  = predict_emotions_batch(texts)
                df["emotions"] = [",".join(e) for e in emotion_lists]
                df["emotion"]  = [e[0] for e in emotion_lists]

                total = len(df)
                sentiment_counts = {
                    "positive": int((df["sentiment"] == "positive").sum()),
                    "neutral":  int((df["sentiment"] == "neutral").sum()),
                    "negative": int((df["sentiment"] == "negative").sum()),
                }
                emotion_counts = emotion_counts_from_lists(emotion_lists)
                comments       = df.to_dict(orient="records")
                for i, c in enumerate(comments):
                    c["emotions"] = emotion_lists[i]
                    c["video_title"] = video["title"]

                all_channel_comments.extend(comments)

                fingerprint = classify_emotional_fingerprint(emotion_counts, total)

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

        # cross-video comparison metrics
        avg_positive = round(sum(r["positive_pct"] for r in results) / len(results), 1)
        avg_negative = round(sum(r["negative_pct"] for r in results) / len(results), 1)
        best_video   = max(results, key=lambda r: r["positive_pct"])
        worst_video  = max(results, key=lambda r: r["negative_pct"])
        fingerprints = [r["fingerprint"]["profile"] for r in results]
        from collections import Counter
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