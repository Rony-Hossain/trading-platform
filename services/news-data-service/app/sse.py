import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from .config import settings

router = APIRouter()

async def _event_stream(redis: Redis, last_id: str | None):
    stream = settings.SSE_REDIS_STREAM
    start_id = last_id or "$"
    while True:
        results = await redis.xread({stream: start_id}, block=15000, count=100)
        if results:
            _, entries = results[0]
            for _id, fields in entries:
                payload = fields.get(b"payload") or fields.get("payload")
                if payload:
                    yield f"id: {_id}\n"
                    yield "event: news\n"
                    yield f"data: {payload.decode() if isinstance(payload, bytes) else payload}\n\n"
                start_id = _id
        else:
            # heartbeat
            yield ": keep-alive\n\n"

@router.get("/stream/news")
async def stream_news(request: Request):
    redis = request.app.state.redis
    last_event_id = request.headers.get("Last-Event-ID")
    return StreamingResponse(_event_stream(redis, last_event_id), media_type="text/event-stream")
