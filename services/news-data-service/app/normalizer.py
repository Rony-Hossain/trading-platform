from datetime import datetime, timezone
from .schema import RawContentIn
from .ccs import sanitize_html, canon_authors, map_categories

async def normalize(p: dict) -> dict:
    r = RawContentIn(**p)
    body = sanitize_html(r.body)
    authors = canon_authors(r.authors)
    cats = map_categories(r.categories)
    return {
        "source": r.source,
        "external_id": r.external_id,
        "url": r.url,
        "title": r.title,
        "body": body,
        "authors": authors,
        "categories": cats,
        "language": r.language,
        "published_at": r.published_at,
        "metadata": r.metadata or {},
    }
