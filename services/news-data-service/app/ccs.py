from bs4 import BeautifulSoup

def sanitize_html(text: str | None) -> str | None:
    if not text:
        return text
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(" ", strip=True)
    except Exception:
        return text

def canon_authors(authors: list[str] | None) -> list[str] | None:
    if not authors:
        return authors
    out = []
    for a in authors:
        a = (a or "").strip()
        if a:
            out.append(a)
    return out or None

CATEGORY_MAP = {
    # example mapping; extend as needed
    "Tech": "technology",
    "Technology": "technology",
}

def map_categories(cats: list[str] | None) -> list[str] | None:
    if not cats:
        return cats
    return [CATEGORY_MAP.get(c, c).lower() for c in cats]
