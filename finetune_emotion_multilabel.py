"""
Fine-tunes paraphrase-multilingual-MiniLM-L12-v2 for MULTI-LABEL,
28-category emotion classification using GoEmotions.

Smaller and faster than XLM-RoBERTa-base while still supporting
50+ languages via the same SentencePiece tokenizer as XLM-R.
Expected training time: ~50-70 minutes on RTX 3050 4GB.

Usage:
    python finetune_emotion_multilabel.py
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from datasets import Dataset

# ── config ──────────────────────────────────────────────────────────────
TRAIN_CSV = "goemotions_train.csv"
VAL_CSV   = "goemotions_val.csv"
TEST_CSV  = "goemotions_test.csv"

# Multilingual MiniLM — 118M params, 50+ languages, AutoTokenizer compatible,
# same SentencePiece tokenizer as XLM-R. Much faster than xlm-roberta-base
# on 4GB VRAM while keeping strong multilingual coverage.
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# New folder — your existing English model stays untouched
OUTPUT_DIR = "./fine-tuned-emotion-model-multilingual"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]
NUM_LABELS = len(EMOTION_LABELS)

PREDICTION_THRESHOLD  = 0.5
NUM_EPOCHS            = 4
BATCH_SIZE            = 16    # back to 16 — model is small enough
GRADIENT_ACCUMULATION = 1
LEARNING_RATE         = 2e-5


# ── data loading ────────────────────────────────────────────────────────

def load_csv_as_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    label_matrix = df[EMOTION_LABELS].values.astype(np.float32)
    return Dataset.from_dict({
        "text":   df["text"].astype(str).tolist(),
        "labels": label_matrix.tolist(),
    })


def compute_pos_weights(train_df: pd.DataFrame) -> torch.Tensor:
    weights = []
    total   = len(train_df)
    for emo in EMOTION_LABELS:
        positive_count = train_df[emo].sum()
        negative_count = total - positive_count
        weight = negative_count / max(positive_count, 1)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


# ── custom trainer with weighted BCE loss ──────────────────────────────

class MultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── metrics ─────────────────────────────────────────────────────────────

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    probs       = torch.sigmoid(torch.tensor(logits))
    predictions = (probs >= PREDICTION_THRESHOLD).int().numpy()
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


# ── main ────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    train_df      = pd.read_csv(TRAIN_CSV)
    train_dataset = load_csv_as_dataset(TRAIN_CSV)
    val_dataset   = load_csv_as_dataset(VAL_CSV)
    test_dataset  = load_csv_as_dataset(TEST_CSV)

    print(f"  train: {len(train_dataset)} examples")
    print(f"  val:   {len(val_dataset)} examples")
    print(f"  test:  {len(test_dataset)} examples")

    print(f"\nLoading tokenizer and base model from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
    model.config.label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}

    print("\nTokenizing...")
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset   = val_dataset.map(tokenize_fn, batched=True)
    test_dataset  = test_dataset.map(tokenize_fn, batched=True)

    print("Computing per-label class weights...")
    pos_weight = compute_pos_weights(train_df)
    for emo, w in zip(EMOTION_LABELS, pos_weight.tolist()):
        print(f"  {emo:15s} weight={w:.1f}")

    training_args = TrainingArguments(
        output_dir="./training-checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    print("\nStarting fine-tuning...")
    print(f"  Model:  {BASE_MODEL}")
    print(f"  Epochs: {NUM_EPOCHS}, batch size: {BATCH_SIZE}, lr: {LEARNING_RATE}")
    print(f"  fp16:   {torch.cuda.is_available()}")
    print("  Expected time: ~50-70 minutes on RTX 3050.\n")

    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Test micro-F1: {test_results['eval_micro_f1']:.3f}")
    print(f"  Test macro-F1: {test_results['eval_macro_f1']:.3f}")

    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nDone! Model saved to {OUTPUT_DIR}")
    print("\nTo activate it, change one line in api.py:")
    print(f'  EMOTION_MODEL_PATH = "{OUTPUT_DIR}"')
    print("Then restart uvicorn.")


if __name__ == "__main__":
    main()