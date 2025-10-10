import os
import httpx
from datetime import datetime, timezone
from .base import AdapterBase

class FinnhubCompanyAdapter(AdapterBase):
    """Company news by symbol via /company-news?symbol=…&from=YYYY-MM-DD&to=YYYY-MM-DD.
    Symbols are pulled from a dynamic WatchlistSource (env list + file + Redis set) with periodic refresh.
    Docs: https://finnhub.io/docs/api/company-news
    """

    def __init__(self, out_queue, circuit, watchlist):
        super().__init__("finnhub-company", out_queue, circuit)
        self.base = "https://finnhub.io/api/v1"
        self.token = os.getenv("FINNHUB_API_KEY", "")
        if not self.token:
            raise RuntimeError("FINNHUB_API_KEY not set")
        self.watchlist = watchlist

    async def fetch_window(self, start, end):
        f = start.date().isoformat()
        t = end.date().isoformat()
        symbols = await self.watchlist.get_symbols()
        if not symbols:
            return
        async with httpx.AsyncClient(timeout=20) as client:
            for sym in symbols:
                params = {"symbol": sym, "from": f, "to": t, "token": self.token}
                r = await client.get(f"{self.base}/company-news", params=params)
                r.raise_for_status()
                items = r.json() or []
                for it in items:
                    dt = it.get("datetime")
                    published_at = datetime.fromtimestamp(dt, tz=timezone.utc) if isinstance(dt, (int, float)) else datetime.now(timezone.utc)
                    yield {
                        "source": "finnhub",
                        "external_id": str(it.get("id") or f"{sym}:{it.get('url')}"),
                        "url": it.get("url") or "",
                        "title": it.get("headline") or "",
                        "body": it.get("summary") or None,
                        "authors": None,
                        "categories": ["company"],
                        "language": "en",
                        "published_at": published_at,
                        "metadata": {"symbol": sym, "source": it.get("source"), "image": it.get("image")},
                    }
