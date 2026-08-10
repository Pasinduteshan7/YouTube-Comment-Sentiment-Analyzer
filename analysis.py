"""
Analysis logic: mixed sentiment detection, topic modelling, emotional
fingerprinting, conflict detection, like-weighted emotions, and pin suggestions.

Extracted from api.py to reduce that file from 880 to ~250 lines.
"""

import re
from models import EMOTION_LABELS


# ── mixed sentiment ────────────────────────────────────────────────────

CONTRAST_WORDS = [
    " but ", " however ", " although ", " though ",
    " yet ", " despite ", " nevertheless ", " while ",
    " whereas ", " even though ", " on the other hand "
]


def detect_mixed_sentiment(text: str, sent_pipeline) -> dict:
    """Detects if a comment expresses conflicting sentiments (e.g. 'Great video, but audio is bad')."""
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


# ── emotion counts ────────────────────────────────────────────────────

def emotion_counts_from_lists(emotion_lists: list) -> dict:
    """Counts emotion occurrences across all comments."""
    counts = {e: 0 for e in EMOTION_LABELS}
    for emo_list in emotion_lists:
        for emo in emo_list:
            if emo in counts:
                counts[emo] += 1
    return counts


# ── topic modelling ────────────────────────────────────────────────────

def detect_topics(comments: list) -> list:
    """Rule-based topic detection using keyword matching."""
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
    """Classifies the overall emotional profile of the comment section."""
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
    """Finds comments that express emotionally conflicting pairs (e.g. joy + sadness)."""
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
    """Calculates emotion scores weighted by comment likes (popular opinions matter more)."""
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


# ── pin suggestions ────────────────────────────────────────────────────

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

    questions = [c for c in comments if "curiosity" in get_emotions(c)]
    best_question = max(questions, key=lambda c: int(c.get("likes", 0)), default=None)

    conflicted = [
        c for c in comments
        if "approval" in get_emotions(c) and "disapproval" in get_emotions(c)
    ]
    best_conflicted = max(conflicted, key=lambda c: int(c.get("likes", 0)), default=None)

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


# ── comment cleaning ──────────────────────────────────────────────────

def clean_comments(df) -> "pd.DataFrame":
    """
    Removes low-quality comments before inference:
    - Pure emoji or symbol-only comments
    - Comments under 5 meaningful characters
    - Exact duplicates
    """
    import pandas as pd

    def is_meaningful(text: str) -> bool:
        text = str(text).strip()
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
