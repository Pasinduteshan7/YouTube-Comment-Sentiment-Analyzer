"""
Model loading and inference for sentiment + emotion analysis.
Centralises all ML model operations so they are consistent across
api.py and analyser.py.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

SENTIMENT_MODEL_PATH = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
EMOTION_MODEL_PATH   = "./fine-tuned-emotion-model-multilingual"
TOXICITY_MODEL_PATH  = "citizenlab/distilbert-base-multilingual-cased-toxicity"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]

EMOTION_THRESHOLD = 0.3

# Module-level state (populated by load_models)
sentiment_pipeline = None
emotion_tokenizer  = None
emotion_model      = None
toxicity_pipeline  = None
device             = None


def load_models():
    """Load sentiment and emotion models into memory. Call once at startup."""
    global sentiment_pipeline, emotion_tokenizer, emotion_model, toxicity_pipeline, device

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

    print("Loading fine-tuned 28-label emotion model...")
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
    emotion_model     = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_PATH
    ).to(device)
    emotion_model.eval()

    print("Loading toxicity model...")
    toxicity_pipeline = pipeline(
        "text-classification",
        model=TOXICITY_MODEL_PATH,
        truncation=True, max_length=512,
        device=device_id,
        top_k=None # Get scores for all labels (toxic vs non-toxic)
    )

    print("Models loaded!")


def predict_emotions_batch(texts: list, batch_size: int = 32) -> list:
    """
    Multi-label emotion prediction using sigmoid thresholding.
    Returns a list of lists, where each inner list contains the detected
    emotion labels for the corresponding input text.
    """
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


def predict_sentiment_batch(texts: list, batch_size: int = 16) -> list:
    """
    Runs sentiment analysis and returns list of dicts with 'label' and 'score'.
    """
    results = sentiment_pipeline(texts, batch_size=batch_size, truncation=True, max_length=512)
    for r in results:
        r["label"] = r["label"].lower()
    return results


def get_sentiment_pipeline():
    """Returns the loaded sentiment pipeline for use in mixed sentiment detection."""
    return sentiment_pipeline


def predict_toxicity_batch(texts: list, batch_size: int = 16) -> list:
    """
    Runs toxicity classification.
    Returns a list of dicts with 'is_toxic' (bool) and 'toxicity_score' (float).
    """
    results = toxicity_pipeline(texts, batch_size=batch_size, truncation=True, max_length=512)
    out = []
    # `results` is a list of lists because top_k=None returns all label scores
    for res_list in results:
        toxic_score = 0.0
        for label_score in res_list:
            if label_score["label"] == "toxic":
                toxic_score = label_score["score"]
                break
        out.append({
            "is_toxic": toxic_score >= 0.985,
            "toxicity_score": toxic_score
        })
    return out
