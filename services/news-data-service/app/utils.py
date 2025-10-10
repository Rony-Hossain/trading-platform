import hashlib
import urllib.parse

def host_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""

def exact_dedupe_key(title: str, url: str) -> str:
    s = (title or "").lower() + "|" + host_of(url)
    return hashlib.sha256(s.encode()).hexdigest()
