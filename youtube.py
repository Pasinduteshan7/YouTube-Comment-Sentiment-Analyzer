"""
YouTube API helpers: video info retrieval, channel video listing.
"""

import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()


def get_video_id(url: str) -> str:
    """Extracts video ID from a YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url.split("/")[-1].split("?")[0]


def get_video_info(video_id: str) -> dict:
    """Retrieves metadata (title, channel, stats) for a YouTube video."""
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


def fetch_channel_videos(channel_url: str, max_videos: int = 10) -> list:
    """
    Given a channel URL or playlist URL, returns a list of dicts:
    [{"video_id": "...", "title": "...", "thumbnail": "...", "published": "...", "url": "..."}]
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)

    # resolve channel URL to uploads playlist ID
    if "@" in channel_url or "channel/" in channel_url:
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
