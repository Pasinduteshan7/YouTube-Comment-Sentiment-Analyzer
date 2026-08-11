import os
import re
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()


def validate_video_id(video_id: str) -> bool:
    """Validates that a string looks like a YouTube video ID (11 alphanumeric chars)."""
    return bool(video_id and re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))


def extract_video_id(video_url: str) -> str:
    """Extracts the video ID from various YouTube URL formats."""
    video_id = None
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    elif "shorts/" in video_url:
        video_id = video_url.split("shorts/")[1].split("?")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = video_url.split("/")[-1].split("?")[0]
    return video_id


def fetch_comments(video_url, max_comments=100):
    """
    Fetches YouTube comments for a given video URL.
    Returns a DataFrame with columns: text, author, likes.
    Returns an empty DataFrame on error.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key or api_key == "your_youtube_api_key_here":
        print("Error: YOUTUBE_API_KEY is missing or not configured.")
        print("  1. Get a key from https://console.cloud.google.com/")
        print("  2. Add it to your .env file: YOUTUBE_API_KEY=your_key_here")
        return pd.DataFrame()

    video_id = extract_video_id(video_url)
    if not validate_video_id(video_id):
        print(f"Error: Invalid YouTube video ID '{video_id}' extracted from URL: {video_url}")
        print("  Supported formats:")
        print("    https://www.youtube.com/watch?v=VIDEO_ID")
        print("    https://www.youtube.com/shorts/VIDEO_ID")
        print("    https://youtu.be/VIDEO_ID")
        return pd.DataFrame()

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page = None

    try:
        while len(comments) < max_comments:
            res = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page,
                textFormat="plainText"
            ).execute()

            for item in res["items"]:
                s = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": s["textDisplay"],
                    "author": s["authorDisplayName"],
                    "likes": s["likeCount"],
                    "publishedAt": s["publishedAt"]
                })

            next_page = res.get("nextPageToken")
            if not next_page: break

        df = pd.DataFrame(comments[:max_comments])
        print(f"Fetched {len(df)} comments for video {video_id}")
        return df
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg:
            print(f"Fetch Error: YouTube API access denied. Your API key may be invalid or quota exceeded.")
        elif "404" in error_msg:
            print(f"Fetch Error: Video not found. Check if the video ID '{video_id}' is correct.")
        elif "commentsDisabled" in error_msg:
            print(f"Fetch Error: Comments are disabled on this video.")
        else:
            print(f"Fetch Error: {e}")
        return pd.DataFrame()