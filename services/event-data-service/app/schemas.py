"""Pydantic schemas for Event Data Service."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventBase(BaseModel):
    symbol: str = Field(..., description="Ticker or entity identifier")
    title: str = Field(..., description="Short headline for the event")
    category: str = Field(..., description="Event category e.g. earnings")
    scheduled_at: datetime = Field(..., description="Scheduled time for the event")
    timezone: Optional[str] = Field(default=None, description="IANA timezone string")
    description: Optional[str] = Field(default=None, description="Longer event summary")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    source: Optional[str] = Field(default=None, description="Provider/source identifier")
    external_id: Optional[str] = Field(default=None, description="Provider-supplied unique id")


class EventCreate(EventBase):
    status: str = Field(default="scheduled", description="Lifecycle status")
    impact_score: Optional[int] = Field(default=None, ge=1, le=10, description="Market-moving potential (1-10)")


class EventUpdate(BaseModel):
    symbol: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    timezone: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    external_id: Optional[str] = None
    status: Optional[str] = None
    impact_score: Optional[int] = Field(default=None, ge=1, le=10)


class EventHeadline(BaseModel):
    id: str
    symbol: str
    headline: str
    summary: Optional[str]
    url: Optional[str]
    published_at: datetime
    source: Optional[str]
    external_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    event_id: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EventHeadlineCreate(BaseModel):
    symbol: str
    headline: str
    published_at: datetime
    summary: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    external_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    event_id: Optional[str] = None


class Event(EventBase):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    impact_score: Optional[int]

    model_config = ConfigDict(from_attributes=True)


class ImpactScoreUpdate(BaseModel):
    impact_score: int = Field(..., ge=1, le=10, description="Impact score between 1 and 10")


class ImpactScoreResponse(BaseModel):
    event_id: str
    impact_score: int
