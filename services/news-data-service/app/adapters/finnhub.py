import os
import httpx
from datetime import datetime, timezone
from .base import AdapterBase

class FinnhubAdapter(AdapterBase):
    """Ingest Finnhub market news via /news?category=… using minId watermark.
    Docs: https://finnhub.io/docs/api (Get latest market news → /news?category=general; supports minId)
    """

    def __init__(self, out_queue, circuit, category: str = "general"):
        super().__init__("finnhub", out_queue, circuit)
        self.base = "https://finnhub.io/api/v1"
        self.token = os.getenv("FINNHUB_API_KEY", "")
        if not self.token:
            raise RuntimeError("FINNHUB_API_KEY not set")
        self.category = category
        self.min_id = 0  # Finnhub 'id' watermark

    async def fetch_window(self, start, end):
        # Finnhub market news doesn't use time windows for /news; it uses minId paging.
        # We still respect the adapter's cadence and safety lag by calling once per tick.
        params = {"category": self.category, "token": self.token}
        if self.min_id:
            params["minId"] = self.min_id
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{self.base}/news", params=params)
            r.raise_for_status()
            items = r.json() or []
        # Track max id as new watermark
        max_id = self.min_id
        for it in items:
            try:
                fid = int(it.get("id") or 0)
                if fid > max_id:
                    max_id = fid
            except Exception:
                pass
            dt = it.get("datetime")
            published_at = datetime.fromtimestamp(dt, tz=timezone.utc) if isinstance(dt, (int, float)) else datetime.now(timezone.utc)
            yield {
                "source": "finnhub",
                "external_id": str(it.get("id") or it.get("url")),
                "url": it.get("url") or "",
                "title": it.get("headline") or "",
                "body": it.get("summary") or None,
                "authors": None,
                "categories": [self.category],
                "language": "en",  # Finnhub market news is predominantly English
                "published_at": published_at,
                "metadata": {"source": it.get("source"), "image": it.get("image")},
            }
        self.min_id = max_id
