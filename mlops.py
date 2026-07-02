import mlflow
import mlflow.sklearn
import pandas as pd
import json
from datetime import datetime
import schedule
import time
import subprocess

# ── point to the same db the UI is using ──
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("youtube-sentiment-analyser")

def log_analysis_run(csv_path="comments_analysed.csv"):
    """Log every analysis run to MLflow so you can track results over time."""
    df = pd.read_csv(csv_path)

    # calculate metrics
    total     = len(df)
    pos_pct   = round(len(df[df.sentiment == "positive"]) / total * 100, 2)
    neg_pct   = round(len(df[df.sentiment == "negative"]) / total * 100, 2)
    neu_pct   = round(len(df[df.sentiment == "neutral"])  / total * 100, 2)
    top_emotion = df["emotion"].value_counts().idxmax()
    avg_sent_score = round(df["sentiment_score"].mean(), 3)
    avg_emo_score  = round(df["emotion_score"].mean(), 3)

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # log parameters
        mlflow.log_param("total_comments",    total)
        mlflow.log_param("sentiment_model",   "cardiffnlp/twitter-roberta-base-sentiment-latest")
        mlflow.log_param("emotion_model",     "j-hartmann/emotion-english-distilroberta-base")
        mlflow.log_param("analysis_date",     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # log metrics
        mlflow.log_metric("positive_pct",        pos_pct)
        mlflow.log_metric("negative_pct",        neg_pct)
        mlflow.log_metric("neutral_pct",         neu_pct)
        mlflow.log_metric("avg_sentiment_score", avg_sent_score)
        mlflow.log_metric("avg_emotion_score",   avg_emo_score)

        # log emotion counts
        for emotion, count in df["emotion"].value_counts().items():
            mlflow.log_metric(f"emotion_{emotion}", int(count))

        # log the csv as artifact
        mlflow.log_artifact(csv_path)

        print(f"\n MLflow run logged successfully!")
        print(f"   Total comments   : {total}")
        print(f"   Positive         : {pos_pct}%")
        print(f"   Negative         : {neg_pct}%")
        print(f"   Neutral          : {neu_pct}%")
        print(f"   Top emotion      : {top_emotion}")
        print(f"   Avg sent score   : {avg_sent_score}")

    return {
        "total": total, "positive_pct": pos_pct,
        "negative_pct": neg_pct, "neutral_pct": neu_pct,
        "top_emotion": top_emotion
    }


def scheduled_pipeline(video_url, interval_hours=24):
    """Automatically re-fetch and re-analyse a video every N hours."""

    def run():
        print(f"\n[{datetime.now()}] Running scheduled analysis...")
        subprocess.run(["python", "fetcher.py"])
        subprocess.run(["python", "analyser.py"])
        log_analysis_run()
        print(f"Next run in {interval_hours} hours.")

    run()  # run immediately first
    schedule.every(interval_hours).hours.do(run)

    print(f"\nScheduler running — will re-analyse every {interval_hours} hours.")
    print("Press Ctrl+C to stop.\n")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    print("Logging current results to MLflow...")
    log_analysis_run()