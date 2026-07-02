import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def fetch_comments(video_url, max_comments=100):
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)

    # Extract ID
    video_id = None
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    elif "shorts/" in video_url:
        video_id = video_url.split("shorts/")[1].split("?")[0]
    else:
        video_id = video_url.split("/")[-1].split("?")[0]

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
                    "likes": s["likeCount"]
                })

            next_page = res.get("nextPageToken")
            if not next_page: break

        df = pd.DataFrame(comments[:max_comments])
        return df
    except Exception as e:
        print(f"Fetch Error: {e}")
        return pd.DataFrame()