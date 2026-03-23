import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def fetch_comments(video_url, max_comments=200):
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)

    # extract video ID from URL
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = video_url.split("/")[-1]

    comments, next_page = [], None

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
                "text":   s["textDisplay"],
                "likes":  s["likeCount"],
                "author": s["authorDisplayName"],
                "date":   s["publishedAt"]
            })

        next_page = res.get("nextPageToken")
        if not next_page:
            break

    df = pd.DataFrame(comments[:max_comments])
    df.to_csv("comments.csv", index=False)
    print(f"Done! {len(df)} comments saved to comments.csv")
    return df

# --- test it with any YouTube URL ---
if __name__ == "__main__":
    df = fetch_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(df.head())