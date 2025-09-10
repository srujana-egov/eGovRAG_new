# src/ingest.py
import os
import glob
import re
import json
import datetime
import hashlib
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from .utils import get_env_var, insert_chunk, count_tokens, chunk_exists
from .retrieval import get_embedding

# =========================
# Tunables (sane defaults)
# =========================
MAX_EMBEDDING_TOKENS = 8191

# ~600-token chunks with ~120-token overlap works well for modern embedders
CFG_TARGET_TOKENS = get_required_env("CHUNK_TARGET_TOKENS", cast=int)
CFG_OVERLAP_TOKENS = get_required_env("CHUNK_OVERLAP_TOKENS", cast=int)

# Filter and safety rails
CFG_MIN_CHARS = get_required_env("CHUNK_MIN_CHARS", cast=int)
CFG_MAX_CHARS = get_required_env("CHUNK_MAX_CHARS", cast=int)

# Atomic “action” lines (turned OFF by default; they polluted recall)
CFG_ENABLE_ATOMICS = get_required_env("CHUNK_ENABLE_ATOMICS", cast=lambda s: s.lower() == "true")

# -------------------------------------------------
# Important/action lines (only used if atomics on)
# -------------------------------------------------
IMPORTANT_KEYWORDS = [
  "Harry Potter",
  "The Boy Who Lived",
  "Privet Drive",
  "Dursleys (Vernon, Petunia, Dudley)",
  "Muggles",
  "Hogwarts",
  "Hagrid (Rubeus Hagrid)",
  "Albus Dumbledore",
  "Minerva McGonagall",
  "Professor Snape (Severus Snape)",
  "Professor Quirrell",
  "Voldemort (You-Know-Who)",
  "Sorcerer’s Stone (Philosopher’s Stone)",
  "Nicolas Flamel",
  "Gringotts",
  "Diagon Alley",
  "Ollivanders",
  "The Leaky Cauldron",
  "Platform Nine and Three-Quarters",
  "The Sorting Hat",
  "Gryffindor, Slytherin, Ravenclaw, Hufflepuff",
  "Ron Weasley, Hermione Granger",
  "Draco Malfoy",
  "Neville Longbottom",
  "Nearly Headless Nick",
  "Peeves",
  "Filch (Argus Filch) & Mrs. Norris",
  "Quidditch",
  "Golden Snitch, Bludger, Quaffle",
  "Nimbus Two Thousand",
  "Invisibility Cloak",
  "Mirror of Erised",
  "Fluffy (three-headed dog)",
  "Devil’s Snare",
  "Wizard’s Chess",
  "Troll (mountain troll)",
  "Forbidden Forest",
  "Centaurs (Firenze, Bane, Ronan)",
  "Unicorn blood",
  "The Philosopher’s/Sorcerer’s Stone protection trials",
  "The scar (lightning bolt)",
  "Godric’s Hollow (referenced)",
  "The Dursleys’ cupboard under the stairs",
  "The Hogwarts Express",
  "Hedwig",
  "The Dueling/“Midnight Duel” incident",
  "The Letters from No One (inundation of Hogwarts letters)",
  "The Vanishing Glass (zoo snake)",
  "McGonagall’s Transfiguration, Flitwick’s Charms, Sprout’s Herbology, Snape’s Potions",
  "The Gringotts break-in (vault 713)",
  "The Put-Outer (Deluminator)",
  "The House Cup",
  "The Rememberall (Neville’s Remembrall)",
  "The Owl Post",
  "The Deathly green light (Avada Kedavra implied)"
]

IMPORTANT_RE = re.compile("|".join(re.escape(k) for k in IMPORTANT_KEYWORDS), re.IGNORECASE)

def extract_important_lines(chunk: str) -> List[str]:
    if not chunk:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?:\n|^)\s*•\s*", chunk)
    out, seen = [], set()
    for p in parts:
        s = (p or "").strip()
        if not s or len(s) < 80 or len(s) > 260 or s.endswith("?"):
            continue
        if IMPORTANT_RE.search(s):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out

# -----------------
# Markdown parsing
# -----------------
# ✅ treat CHAPTER as a heading
MD_HEADING_RE = re.compile(r"^(#{1,6}|CHAPTER)\s+(.*)$", re.IGNORECASE)
MD_FM_BOUNDARY = re.compile(r"^---\s*$")

def _stable_doc_id(base_name: str, chunk_index: int, mtime_ns: Opt7 bhjm ional[int]) -> str:
    suffix = str(mtime_ns) if mtime_ns is not None else "na"
    return f"{base_name}:{suffix}:{chunk_index}"

def _file_metadata(path: str) -> Dict:
    try:
        stat = os.stat(path)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        return {"filesize": stat.st_size, "modified": mtime}
    except Exception:
        return {}

def _token_cap(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        if count_tokens(candidate) <= max_tokens:
            lo = mid + 1
        else:
            hi = mid
    trimmed = " ".join(words[:max(lo-1, 1)]).rstrip()
    return (trimmed + "…") if trimmed else text[:200] + "…"

def _strip_md_noise(md: str) -> str:
    lines = md.splitlines()
    i = 0
    if i < len(lines) and MD_FM_BOUNDARY.match(lines[i]):
        i += 1
        while i < len(lines) and not MD_FM_BOUNDARY.match(lines[i]):
            i += 1
        i = min(i + 1, len(lines))
    body = "\n".join(lines[i:])
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"`[^`]+`", " ", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", r"\1 \2", body)
    body = re.sub(r"^\s*[\-\*\+]\s+", "• ", body, flags=re.MULTILINE)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()

def _parse_markdown_sections(md_text: str):
    clean = _strip_md_noise(md_text)
    lines = clean.splitlines()
    sections, stack, buf = [], [], []
    current = {"level": 0, "title": "", "path": "", "content": ""}

    def push_section():
        nonlocal buf, current
        content = "\n".join(buf).strip()
        if content:
            sec = dict(current)
            sec["content"] = content
            sections.append(sec)
        buf = []

    for raw in lines:
        h = MD_HEADING_RE.match(raw)
        if h:
            push_section()
            lvl = len(h.group(1)) if h.group(1).startswith("#") else 1
            title = h.group(2).strip().rstrip("#").strip()
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
            path = " > ".join(t for _, t in stack)
            current = {"level": lvl, "title": title, "path": path, "content": ""}
            continue
        buf.append(raw)
    push_section()
    return sections

# -----------------
# Chunking logic
# -----------------
# pip install langchain langchain-openai tiktoken
# Optional: pip install spacy && python -m spacy download en_core_web_sm

from typing import List, Iterable
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# 1) Semantic boundaries: sentence-level semantic grouping
def semantic_boundaries(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    breakpoint_threshold_type: str = "percentile",   # "percentile" or "standard_deviation"
    breakpoint_threshold_amount: float = 95,         # e.g., 95th percentile; tune as needed
    buffer_size: int = 1                              # local context window for grouping
) -> List[str]:
    """
    Use LangChain SemanticChunker to split text into semantically coherent spans.
    Returns variable-length spans without hard token caps (to be enforced in step 2).
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        buffer_size=buffer_size,
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]

# 2) Hard token cap with overlap: enforce embedding-window limits
def token_hard_cap_with_overlap(
    spans: Iterable[str],
    max_tokens: int = 8191,           # model hard limit for embeddings
    target_chunk_tokens: int = 512,    # desired chunk size inside the hard limit
    chunk_overlap_tokens: int = 64,    # overlap to preserve context across cuts
    encoding_name: str = "cl100k_base" # tiktoken encoding compatible with OpenAI embeddings
) -> List[str]:
    """
    For each semantic span, apply a token-aware recursive splitter to ensure
    chunks stay under max_tokens, using target_chunk_tokens and overlap.
    """
    # Build a token-length function for RecursiveCharacterTextSplitter
    enc = tiktoken.get_encoding(encoding_name)
    def token_len(s: str) -> int:
        return len(enc.encode(s))

    # RecursiveCharacterTextSplitter can operate in token space by providing length_function
    # and choosing conservative separators to preserve structure when possible.
    size_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=target_chunk_tokens,
        chunk_overlap=chunk_overlap_tokens,
        # separators defaults are fine; can pass custom separators if desired
    )

    hard_capped_chunks: List[str] = []
    for span in spans:
        # First, if a span already fits under max_tokens, keep it as-is (still may be > target size)
        if token_len(span) <= max_tokens:
            # If it exceeds target size, split; otherwise keep
            if token_len(span) > target_chunk_tokens:
                pieces = size_splitter.split_text(span)
                # Ensure each piece respects max_tokens (rarely violated with the settings above)
                for p in pieces:
                    if token_len(p) <= max_tokens:
                        hard_capped_chunks.append(p)
                    else:
                        # Fallback: strict slicing in token space
                        toks = enc.encode(p)
                        for i in range(0, len(toks), max_tokens):
                            hard_capped_chunks.append(enc.decode(toks[i:i+max_tokens]))
            else:
                hard_capped_chunks.append(span)
        else:
            # Span is oversized: strictly slice under max_tokens, then optionally re-window to target
            toks = enc.encode(span)
            slices = [enc.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]
            # Optionally re-window each strict slice down to target size (keeps overlap behavior consistent)
            for s in slices:
                if token_len(s) > target_chunk_tokens:
                    pieces = size_splitter.split_text(s)
                    for p in pieces:
                        hard_capped_chunks.append(p if token_len(p) <= max_tokens else enc.decode(enc.encode(p)[:max_tokens]))
                else:
                    hard_capped_chunks.append(s)

    return hard_capped_chunks

# -----------------
# Dedup helpers
# -----------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _hash_text(text: str) -> str:
    return hashlib.md5(_norm(text).encode("utf-8")).hexdigest()

_seen_hashes = set()

# =========================
# Markdown ingestion
# =========================
def ingest_md_to_pg(md_path: str, get_embedding_fn, crawled: Optional[set] = None, depth: int = 0):
    base_name = os.path.basename(md_path)

    # ✅ reset dedup per file
    global _seen_hashes
    _seen_hashes.clear()

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
    except Exception as e:
        return f"❌ Error reading {md_path}: {e}"

    sections = _parse_markdown_sections(md_text)
    if not sections:
        return f"⚠️ No sections found in {base_name}"

    try:
        st = os.stat(md_path)
        mtime_ns = getattr(st, "st_mtime_ns", None)
        mtime_str = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
        file_meta = _file_metadata(md_path)
    except Exception:
        mtime_ns = None
        mtime_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_meta = {}

    meta_sidecar: Dict = {}
    sidecar_path = os.path.splitext(md_path) + ".meta.json"
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                meta_sidecar = json.load(f) or {}
        except Exception:
            meta_sidecar = {}

    chunk_counter = 0
    for sec in sections:
        # NEW: semantic first, then token-capped windows with overlap
        spans = semantic_boundaries(
            sec["content"],
            embedding_model="text-embedding-3-small",              # tune if needed
            breakpoint_threshold_type="percentile",                 # or "standard_deviation"
            breakpoint_threshold_amount=95,                         # 90–97 typical
            buffer_size=1,                                          # 1–3 typical
        )
        sec_chunks = token_hard_cap_with_overlap(
            spans,
            max_tokens=MAX_EMBEDDING_TOKENS,
            target_chunk_tokens=CFG_TARGET_TOKENS,
            chunk_overlap_tokens=CFG_OVERLAP_TOKENS,
            encoding_name="cl100k_base",                            # match embedder tokenizer
        )

        for ch in sec_chunks:
            if not ch.strip() or len(ch) < CFG_MIN_CHARS:
                continue

            prefix = sec["path"].strip()
            text = f"{prefix} — {ch}" if prefix else ch

            # Final guardrail against any possible overshoot
            if count_tokens(text) > MAX_EMBEDDING_TOKENS:
                text = _token_cap(text, MAX_EMBEDDING_TOKENS - 32)

            h = _hash_text(text)
            if h in _seen_hashes:
                continue
            _seen_hashes.add(h)

            doc_id = _stable_doc_id(base_name, chunk_counter, mtime_ns)
            chunk_counter += 1

            metadata = {
                "source": meta_sidecar.get("source") or "markdown",
                "product": meta_sidecar.get("product") or "HCM",
                "doc_type": "Docs/Markdown",
                "filename": base_name,
                "abs_path": os.path.abspath(md_path),
                "section_level": sec["level"],
                "section_title": sec["title"],
                "section_path": sec["path"],
                "url": meta_sidecar.get("url", ""),
                "page_title": meta_sidecar.get("title", ""),
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_modified": mtime_str,
                **file_meta,
            }

            if not chunk_exists(doc_id):
                insert_chunk(doc_id, text, metadata, get_embedding_fn)
                print(f"✅ Ingested MD chunk {doc_id} ({sec['path']})")
            else:
                print(f"⏩ Skipping existing MD chunk {doc_id}")

            if CFG_ENABLE_ATOMICS:
                for idx, line in enumerate(extract_important_lines(text)):
                    h_line = _hash_text(line)
                    if h_line in _seen_hashes:
                        continue
                    _seen_hashes.add(h_line)
                    special_id = f"{doc_id}:important:{idx}"
                    if not chunk_exists(special_id):
                        insert_chunk(special_id, line, metadata, get_embedding_fn)
                        print(f"⭐ Ingested atomic MD {special_id}")

    return f"✅ Finished {base_name}"

# =========================
# Batch ingestion
# =========================
def ingest_all(folder_path: str, get_embedding_fn):
    md_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
    print(f"📂 Found {len(md_files)} MD files in {folder_path} (recursive)")
    results = []
    for m in md_files:
        try:
            res = ingest_md_to_pg(m, get_embedding_fn, crawled=set(), depth=0)
            results.append(res)
            print(res)
        except Exception as e:
            print(f"❌ Exception (MD) in {m}: {e}")
    print("🎉 All MD docs ingested.")
    return results

if __name__ == "__main__":
    folder = get_env_var("INGEST_MD_DIR", "data")
    ingest_all(folder, get_embedding)
