# src/ingest_web.py
import os
import re
import time
import json
import queue
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .utils import get_env_var
from .ingest import ingest_all  # reuse your MD pipeline

load_dotenv()

ROOT_URL     = get_env_var("HEALTH_ROOT_URL", "https://docs.digit.org/health")
OUT_DIR      = get_env_var("HEALTH_OUT_DIR", "data/gitbook_pages")
ALLOWED_HOST = get_env_var("HEALTH_ALLOWED_HOST", "docs.digit.org")
MAX_PAGES    = int(get_env_var("HEALTH_MAX_PAGES", "500"))
MAX_DEPTH    = int(get_env_var("HEALTH_MAX_DEPTH", "3"))
FETCH_DELAY  = float(get_env_var("HEALTH_FETCH_DELAY_S", "0.15"))
TIMEOUT      = int(get_env_var("HEALTH_HTTP_TIMEOUT", "10"))

HEADERS = {"User-Agent": "RAG-Ingest/1.0 (+md-crawler)"}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s\-\/]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = s.strip("-")
    return s or "page"

def path_slug(url: str) -> str:
    p = urlparse(url)
    path = p.path.rstrip("/")
    if not path:
        return "index"
    return slugify(path.lstrip("/"))

def is_allowed(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc != ALLOWED_HOST:
        return False
    # Keep only /health/**
    return p.path.startswith("/health")

def html_to_markdown_like(html: str) -> tuple[str, dict]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body or soup

    parts = []
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()

    def tsv_from_table(table):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append("\t".join(cells))
        return "\n".join(rows)

    for tag in main.descendants:
        if not getattr(tag, "name", None):
            continue
        name = tag.name.lower()

        if name in ("h1","h2","h3"):
            level = int(name[1])
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append("\n" + "#"*level + " " + txt + "\n")

        elif name == "p":
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append(txt + "\n")

        elif name == "li":
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append("• " + txt + "\n")

        elif name == "table":
            tsv = tsv_from_table(tag)
            if tsv:
                parts.append("\n" + tsv + "\n")

        elif name in ("nav","header","footer","script","style"):
            continue

    text = "\n".join(p for p in parts if p and p.strip())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, {"title": title}

def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        if "404" in r.url and url != r.url:
            return None
        return r.text
    except Exception:
        return None

def extract_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("#"):
            continue
        abs_url = urljoin(base_url, href)
        if is_allowed(abs_url):
            out.append(abs_url.split("#", 1)[0])
    return list(dict.fromkeys(out))

# ---------- crawler ----------

def crawl_and_write(start_url: str):
    seen = set()
    q = queue.Queue()
    q.put((start_url, 0))

    written = 0

    while not q.empty() and written < MAX_PAGES:
        url, depth = q.get()
        if url in seen:
            continue
        seen.add(url)
        if depth > MAX_DEPTH:
            continue

        time.sleep(FETCH_DELAY)
        html = fetch(url)
        if not html:
            continue

        text, meta = html_to_markdown_like(html)
        if not text.strip():
            continue

        slug = path_slug(url)
        md_path   = os.path.join(OUT_DIR, f"{slug}.md")
        meta_path = os.path.join(OUT_DIR, f"{slug}.meta.json")

        # ✅ Ensure parent directories exist
        os.makedirs(os.path.dirname(md_path), exist_ok=True)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {meta.get('title') or slug}\n\n")
            f.write(text.strip() + "\n")

        meta_payload = {
            "url": url,
            "title": meta.get("title") or slug,
            "source": "gitbook",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        print(f"📝 Saved {url} -> {md_path}")
        written += 1

        for nxt in extract_links(url, html):
            if nxt not in seen:
                q.put((nxt, depth + 1))

    print(f"✅ Crawl finished: {written} pages written to {OUT_DIR}")

def run():
    crawl_and_write(ROOT_URL)
    from .retrieval import get_embedding
    print("📥 Ingesting crawled Markdown pages...")
    ingest_all(OUT_DIR, get_embedding)

if __name__ == "__main__":
    run()
