"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    url: str
    max_comments: int = 100


class ChannelAnalysisRequest(BaseModel):
    url: str
    max_videos: int = 5
    comments_per_video: int = 100
