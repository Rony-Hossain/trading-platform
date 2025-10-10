from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class RawContentIn(BaseModel):
    source: Literal["finnhub","alpha_vantage","newsapi","rss","wire","sec","other"]
    external_id: str
    url: str
    title: str
    body: Optional[str] = None
    authors: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    language: Optional[str] = None
    published_at: datetime
    metadata: Optional[dict] = None

class ContentCreated(BaseModel):
    event: Literal["content.created"] = "content.created"
    schema_version: Literal["v1"] = "v1"  # ✅ Added versioning
    type: Literal["news"] = "news"
    content_id: str
    source: str
    external_id: str
    url: str
    title: str
    body_present: bool
    language: Optional[str]
    published_at: datetime
    as_of_ts: datetime
    ingested_at: datetime
    revision_seq: int
    metadata: Optional[dict]

class ContentCorrected(BaseModel):
    event: Literal["content.corrected"] = "content.corrected"
    schema_version: Literal["v1"] = "v1"  # ✅ Added versioning
    type: Literal["news"] = "news"
    content_id: str
    revision_seq: int
    published_at: datetime
    as_of_ts: datetime

class ContentRetracted(BaseModel):
    event: Literal["content.retracted"] = "content.retracted"
    schema_version: Literal["v1"] = "v1"  # ✅ Added versioning
    type: Literal["news"] = "news"
    content_id: str
    reason: Optional[str]
    as_of_ts: datetime
