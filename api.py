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

@app.on_event("startup")
async def startup_event():
    load_models()


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
) -> list:
    total   = max(len(comments), 1)
    neg_pct = round(sentiment_counts["negative"] / total * 100)
    pos_pct = round(sentiment_counts["positive"] / total * 100)

    def pct(emo):
        return round(emotion_counts.get(emo, 0) / total * 100, 1)

    negative_text = " ".join(
        c["text"] for c in comments if c.get("sentiment") == "negative"
    )[:3000].lower()
    positive_text = " ".join(
        c["text"] for c in comments if c.get("sentiment") == "positive"
    )[:2000].lower()

    suggestions = []

    # 1. Emotional fingerprint
    if fingerprint:
        profile_type = {
            "Creator Loyalty":          "success",
            "Mind-Opening":             "success",
            "Entertainment Hit":        "success",
            "Skill Showcase":           "success",
            "Motivational":             "success",
            "High Energy":              "success",
            "Emotional Storytelling":   "info",
            "Tutorial / Explainer":     "info",
            "Mixed Reception":          "info",
            "Controversial / Divisive": "warning",
        }.get(fingerprint["profile"], "info")
        suggestions.append({
            "type":   profile_type,
            "title":  f"Emotional profile: {fingerprint['profile']}",
            "detail": fingerprint["description"],
        })

    # 2. Like-weighted emotion insight
    if like_weighted:
        top_weighted = list(like_weighted.keys())[:3]
        if top_weighted:
            suggestions.append({
                "type":   "info",
                "title":  "What your most-liked comments feel",
                "detail": (
                    f"Weighting emotions by likes, your engaged audience primarily feels: "
                    f"{', '.join(top_weighted)}. These are the emotions that resonate with "
                    f"viewers who care enough to interact -- prioritise them in future videos."
                ),
            })

    # 3. Sentiment overview
    if neg_pct > 30:
        suggestions.append({
            "type":   "warning",
            "title":  "High negative sentiment detected",
            "detail": f"{neg_pct}% of comments are negative. See the negative emotion breakdown below for specifics.",
        })
    if neg_pct <= 15:
        suggestions.append({
            "type":   "success",
            "title":  "Excellent audience reception",
            "detail": f"Only {neg_pct}% negative comments. Your audience is highly satisfied -- keep this content style.",
        })
    if pos_pct >= 40:
        suggestions.append({
            "type":   "success",
            "title":  "Strong positive engagement",
            "detail": f"{pos_pct}% positive comments. Consider making more content in this topic area.",
        })

    # 4. Negative emotion breakdown
    neg_emotions = {
        "anger":          pct("anger"),
        "annoyance":      pct("annoyance"),
        "disappointment": pct("disappointment"),
        "disgust":        pct("disgust"),
        "disapproval":    pct("disapproval"),
        "fear":           pct("fear"),
        "confusion":      pct("confusion"),
        "sadness":        pct("sadness"),
    }
    dominant_negatives = {k: v for k, v in neg_emotions.items() if v >= 3}
    if dominant_negatives:
        top_neg = max(dominant_negatives, key=dominant_negatives.get)
        neg_advice = {
            "anger":          "This usually signals controversial content or a strong expectation mismatch -- review your thumbnail and title.",
            "annoyance":      "Viewers found something irritating -- common causes are slow pacing, long intros, or repetitive content.",
            "disappointment": "Viewers expected more -- the concept may have been stronger than the execution. Consider a follow-up that delivers more deeply.",
            "disgust":        "Something in this video strongly repelled viewers. Review comments for specific triggers.",
            "disapproval":    "Viewers disagree with a position or decision in this content. Consider addressing it directly in a response or follow-up.",
            "fear":           "Content felt unsettling or threatening to viewers. Ensure the tone matches audience expectations.",
            "confusion":      "Viewers struggled to follow the content. Add chapter markers, clearer structure, or a summary at the end.",
            "sadness":        "Viewers felt sad -- this can be intentional (emotional storytelling) or unintended (bad news, tone mismatch).",
        }
        breakdown_str = ", ".join(
            f"{k} {v}%" for k, v in sorted(dominant_negatives.items(), key=lambda x: -x[1])
        )
        suggestions.append({
            "type":   "warning",
            "title":  f"Negative emotion breakdown -- primary: {top_neg}",
            "detail": f"Negative feelings split as: {breakdown_str}. {neg_advice.get(top_neg, '')}",
        })

    # 5. Gratitude signal
    gratitude_pct = pct("gratitude")
    if gratitude_pct >= 8:
        suggestions.append({
            "type":   "success",
            "title":  f"High gratitude signal ({gratitude_pct}% of comments)",
            "detail": (
                "Viewers feel genuinely thankful for this content -- the strongest predictor of "
                "long-term subscribers. These viewers feel the content gave them something valuable."
            ),
        })
    elif gratitude_pct >= 3:
        suggestions.append({
            "type":   "info",
            "title":  f"Moderate gratitude ({gratitude_pct}% of comments)",
            "detail": (
                "Some viewers feel grateful, but there is room to grow. Try being more explicit "
                "about the value you are delivering -- tutorials, personal stories, and "
                "resource-sharing tend to drive gratitude higher."
            ),
        })

    # 6. Realization signal
    realization_pct = pct("realization")
    if realization_pct >= 10:
        suggestions.append({
            "type":   "success",
            "title":  f"Strong realization signal ({realization_pct}% of comments)",
            "detail": (
                "This video genuinely changed how viewers think or see something. "
                "Content that produces realization has very high share rates and long-term recall -- "
                "consider making more in this format."
            ),
        })
    elif realization_pct >= 5:
        suggestions.append({
            "type":   "info",
            "title":  f"Realization present ({realization_pct}% of comments)",
            "detail": (
                "Some viewers had an aha moment watching this. To amplify this, be more deliberate "
                "about your reveal structure -- build tension before the insight lands."
            ),
        })

    # 7. Curiosity and confusion
    curiosity_pct = pct("curiosity")
    confusion_pct = pct("confusion")
    if confusion_pct >= 10:
        suggestions.append({
            "type":   "warning",
            "title":  f"Confusion is elevated ({confusion_pct}% of comments)",
            "detail": (
                "A significant share of viewers found this content hard to follow. "
                "Add a clear intro, use chapter markers, and consider a summary at the end."
            ),
        })
    if curiosity_pct >= 12 and confusion_pct < 8:
        suggestions.append({
            "type":   "info",
            "title":  f"High curiosity ({curiosity_pct}% of comments)",
            "detail": (
                "Viewers are asking questions and wanting to know more -- a strong signal of engaged interest. "
                "Pin a comment answering the top questions, or make a follow-up video addressing them."
            ),
        })

    # 8. Conflicted comments
    if conflicted and len(conflicted) >= 3:
        conflict_pairs_found = list({c["conflict_pair"] for c in conflicted})[:2]
        suggestions.append({
            "type":   "warning",
            "title":  f"{len(conflicted)} comments carry contradictory emotions",
            "detail": (
                f"The most common emotional conflicts are: {', '.join(conflict_pairs_found)}. "
                f"These viewers felt pulled in two directions -- something worked and something did not. "
                f"See the conflicted comments section below for the specific comments."
            ),
        })

    # 9. Keyword tips from negative comments
    if any(w in negative_text for w in ["long", "slow", "boring", "skip", "too long"]):
        suggestions.append({
            "type":   "warning",
            "title":  "Pacing feedback in negative comments",
            "detail": "Negative comments mention length or pacing. Consider tighter editing and chapter markers.",
        })
    if any(w in negative_text for w in ["audio", "sound", "hear", "mic", "volume"]):
        suggestions.append({
            "type":   "warning",
            "title":  "Audio quality complaints",
            "detail": "Viewers are flagging audio issues. Better microphone or post-processing could significantly help.",
        })
    if any(w in negative_text for w in ["clickbait", "mislead", "thumbnail", "title"]):
        suggestions.append({
            "type":   "warning",
            "title":  "Thumbnail / title mismatch",
            "detail": "Viewers feel the thumbnail or title was misleading. Make sure content delivers on the promise.",
        })

    # 10. Positive opportunities
    if any(w in positive_text for w in ["part 2", "more", "next", "series", "continue", "follow up"]):
        suggestions.append({
            "type":   "info",
            "title":  "Viewers want more content",
            "detail": "Positive comments ask for follow-up. A part 2 or series would perform well.",
        })

    optimism_pct = pct("optimism")
    if optimism_pct >= 10:
        suggestions.append({
            "type":   "success",
            "title":  f"Viewers feel optimistic after watching ({optimism_pct}%)",
            "detail": (
                "Content that leaves viewers feeling hopeful is rare and valuable. "
                "Keep the positive, forward-looking tone in future videos."
            ),
        })

    if not suggestions:
        suggestions.append({
            "type":   "info",
            "title":  "Neutral audience response",
            "detail": "Comments are mostly neutral. Try ending videos with a direct question to boost engagement.",
        })

    return suggestions


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
    }


@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    try:
        df = fetch_comments(request.url, max_comments=request.max_comments)
        if df.empty:
            raise HTTPException(status_code=400, detail="No comments found for this video.")

        video_id   = get_video_id(request.url)
        video_info = get_video_info(video_id)
        texts      = df["text"].astype(str).tolist()

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

        fingerprint   = classify_emotional_fingerprint(emotion_counts, total)
        conflicted    = find_conflicted_comments(comments)
        like_weighted = compute_like_weighted_emotions(comments)
        suggestions   = generate_suggestions(comments, sentiment_counts, emotion_counts,
                                             fingerprint, conflicted, like_weighted)
        topics        = detect_topics(comments)

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
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)