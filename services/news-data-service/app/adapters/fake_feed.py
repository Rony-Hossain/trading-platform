# A minimal fake adapter to prove the pipeline; replace with real providers.
import asyncio
from datetime import datetime, timezone, timedelta
from .base import AdapterBase

class FakeAdapter(AdapterBase):
    def __init__(self, out_queue, circuit):
        super().__init__("fake", out_queue, circuit)
        self._counter = 0

    async def fetch_window(self, start: datetime, end: datetime):
        # generate 3 fake items per tick
        for i in range(3):
            self._counter += 1
            yield {
                "source": "rss",
                "external_id": f"fake-{self._counter}",
                "url": f"https://example.com/{self._counter}",
                "title": f"Sample headline {self._counter}",
                "body": None,
                "authors": None,
                "categories": ["Tech"],
                "language": "en",
                "published_at": datetime.now(timezone.utc) - timedelta(seconds=5),
                "metadata": {"gen": True}
            }
        await asyncio.sleep(0.2)
