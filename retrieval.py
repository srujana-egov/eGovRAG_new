# src/retrieval.py
import os, re, math, json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None

# ---------------------------
# Environment
# ---------------------------
def _env(k: str, d: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if (v is not None and str(v).strip() != "") else d

TABLE = _env("HCMBOT_TABLE", "hcmbot_knowledge")
ID_COL, TXT_COL, META_COL = "id", "document", "metadata"
TSV_COL_PRIMARY, TSV_COL_SECOND = "document_tsv", "tsv_en"
EMB_COL = _env("EMBED_COL", "embedding")

HYBRID_ALPHA   = float(_env("HYBRID_ALPHA", "0.35"))     # 0=pure lexical, 1=pure vector
CAND_MULT      = int(_env("RETRIEVE_CAND_MULT", "10"))   # fetch this many * top_k
MAX_SQL_LIMIT  = int(_env("RETRIEVE_SQL_LIMIT", "300"))
MMR_LAMBDA     = float(_env("MMR_LAMBDA", "0.7"))

# Optional filter to restrict retrieval to a specific filename (by metadata)
FILENAME_FILTER = _env("RETRIEVE_FILENAME", None)

EMBED_MODEL    = _env("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM      = int(_env("EMBED_DIM", "1536"))
EMBED_NORMALIZE= _env("EMBED_NORMALIZE", "1") == "1"

# ---------------------------
# Embeddings (guarded)
# ---------------------------
def get_embedding(text: str) -> List[float]:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for embeddings.")
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = resp.data[0].embedding
    if EMBED_NORMALIZE:
        n = math.sqrt(sum(x*x for x in vec)) or 1.0
        vec = [x / n for x in vec]
    return vec

# ---------------------------
# Light text utils
# ---------------------------
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
AP1 = re.compile(r"\b(\w+)'s\b")
AP2 = re.compile(r"\b(\w+)’s\b")

def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.casefold()
    s = AP1.sub(r"\1", s); s = AP2.sub(r"\1", s)
    s = s.replace("—"," ").replace("–"," ").replace("-"," ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _tf_dict(s: str) -> Dict[str, float]:
    toks = _norm_text(s).split()
    if not toks: return {}
    from collections import Counter
    c = Counter(toks); tot = float(sum(c.values()))
    return {t: v/tot for t, v in c.items()} if tot>0 else {}

def _cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    common = set(a) & set(b)
    num = sum(a[t]*b[t] for t in common)
    na = math.sqrt(sum(v*v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v*v for v in b.values())) or 1.0
    return num/(na*nb)

# ---------------------------
# MMR (relevance-biased)
# ---------------------------
def _cand_score(c: Any) -> float:
    try:
        if isinstance(c, dict): s = float(c.get("score", 0.0))
        elif isinstance(c, (list,tuple)) and len(c)>=2: s=float(c[1])
        else: s=0.0
        return 0.0 if (s is None or math.isnan(s)) else s
    except Exception:
        return 0.0

def _cand_vec(c: Any):
    if isinstance(c, dict): return c.get("tfidf") or c.get("vec") or c.get("embedding")
    if isinstance(c, (list,tuple)) and len(c)>=3: return c[2]
    return None

def mmr_select(scored: List[Any], q_vec=None, k: int=10, lambda_: float=MMR_LAMBDA) -> List[Any]:
    n = len(scored)
    if n==0 or k<=0: return []
    k = min(k,n)
    idx = list(range(n))
    idx.sort(key=lambda i: _cand_score(scored[i]), reverse=True)
    selected = [idx[0]]; avail = idx[1:]
    vecs = [_cand_vec(c) for c in scored]
    rels = [_cand_score(c) for c in scored]
    while avail and len(selected)<k:
        best_i, best_val = None, -1e18
        for i in avail:
            rel = rels[i]
            if isinstance(vecs[i], dict) and any(isinstance(vecs[j], dict) for j in selected):
                div = max(_cosine(vecs[i], vecs[j]) for j in selected if isinstance(vecs[j], dict))
            else:
                div = 0.0
            val = lambda_*rel - (1.0-lambda_)*div
            if val>best_val: best_val, best_i = val, i
        selected.append(best_i); avail.remove(best_i)
    return [scored[i] for i in selected]

# ---------------------------
# Postgres helpers
# ---------------------------
def _pg_connect():
    if psycopg2 is None: return None
    params = dict(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT","5432"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE","require"),
        connect_timeout=int(os.getenv("PGCONNECT_TIMEOUT","20")),
        options=os.getenv("PGOPTIONS"),
    )
    if not params["host"] or not params["dbname"] or not params["user"]:
        return None
    try:
        params = {k:v for k,v in params.items() if v is not None}
        return psycopg2.connect(**params)
    except Exception:
        return None

def _has_column(cur, table: str, col: str) -> bool:
    cur.execute("""SELECT 1 FROM information_schema.columns WHERE table_name=%s AND column_name=%s""",(table, col))
    return cur.fetchone() is not None

def _has_extension(cur, ext: str) -> bool:
    try:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname=%s", (ext,))
        return cur.fetchone() is not None
    except Exception:
        return False

# ---------------------------
# Query building
# ---------------------------
def _informative_terms(s: str, max_terms=8) -> List[str]:
    out, seen = [], set()
    for t in _norm_text(s).split():
        if len(t)>=2 and t not in seen:
            out.append(t); seen.add(t)
        if len(out)>=max_terms: break
    return out

def _prefix_tsquery(q: str) -> Optional[str]:
    terms = _informative_terms(q, max_terms=8)
    if not terms: return None
    return " & ".join(f"{t}:*" for t in terms)

def _build_online_tsv_expr():
    # Title/path/url weighted + body lower weight
    title_en = f"setweight(to_tsvector('english', coalesce({META_COL}->>'page_title','')), 'A')"
    path_en  = f"setweight(to_tsvector('english', coalesce({META_COL}->>'section_path','')), 'B')"
    url_en   = f"setweight(to_tsvector('english', coalesce({META_COL}->>'url','')), 'B')"
    body_en  = f"setweight(to_tsvector('english', coalesce({TXT_COL},'')), 'C')"
    return f"({title_en} || {path_en} || {url_en} || {body_en})"

# ---------------------------
# Candidate retrieval
# ---------------------------
def _hybrid_candidates(conn, query: str, need: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        has_tsv_primary = _has_column(cur, TABLE, TSV_COL_PRIMARY)
        has_tsv_second  = _has_column(cur, TABLE, TSV_COL_SECOND)
        has_vector      = _has_column(cur, TABLE, EMB_COL)

        # choose tsvector expr
        tsv_expr = None
        if has_tsv_primary:
            tsv_expr = TSV_COL_PRIMARY
        elif has_tsv_second:
            tsv_expr = TSV_COL_SECOND
        else:
            tsv_expr = _build_online_tsv_expr()  # on-the-fly

        limit_n = min(max(need, 50), MAX_SQL_LIMIT)
        q_prefix = _prefix_tsquery(query)

        # Build optional filename filter clause and params
        filter_clause = ""
        filter_params: List[Any] = []
        if FILENAME_FILTER:
            filter_clause = f" AND {META_COL}->>'filename' = %s"
            filter_params = [FILENAME_FILTER]

        # 1) strong lexical: websearch
        sql_web = f"""
          SELECT
            {ID_COL} AS id, {TXT_COL} AS document, {META_COL} AS metadata,
            ts_rank_cd({tsv_expr}, websearch_to_tsquery('english', %s)) AS ts_score,
            0.0::float AS vec_score
          FROM {TABLE}
          WHERE {tsv_expr} @@ websearch_to_tsquery('english', %s){filter_clause}
          ORDER BY ts_score DESC
          LIMIT %s
        """
        params_web: List[Any] = [query, query] + filter_params + [limit_n]
        cur.execute(sql_web, params_web)
        rows = cur.fetchall()

        # 2) prefix tsquery supplement (only if we have terms)
        if q_prefix:
            sql_pref = f"""
              SELECT
                {ID_COL} AS id, {TXT_COL} AS document, {META_COL} AS metadata,
                ts_rank_cd({tsv_expr}, to_tsquery('simple', %s)) * 0.6 AS ts_score,
                0.0::float AS vec_score
              FROM {TABLE}
              WHERE {tsv_expr} @@ to_tsquery('simple', %s){filter_clause}
              ORDER BY ts_score DESC
              LIMIT %s
            """
            params_pref: List[Any] = [q_prefix, q_prefix] + filter_params + [limit_n//2]
            cur.execute(sql_pref, params_pref)
            rows += cur.fetchall()

        # 3) vector (guarded)
        vrows: List[Dict[str, Any]] = []
        if has_vector:
            try:
                qvec = get_embedding(query)
                sql_vec = f"""
                  SELECT
                    {ID_COL} AS id, {TXT_COL} AS document, {META_COL} AS metadata,
                    0.0::float AS ts_score,
                    (1.0 - ({EMB_COL} <=> %s))::float AS vec_score
                  FROM {TABLE}
                  WHERE (%s IS NOT NULL){' AND ' + META_COL + "->>'filename' = %s" if FILENAME_FILTER else ''}
                  ORDER BY {EMB_COL} <=> %s
                  LIMIT %s
                """
                if FILENAME_FILTER:
                    params_vec: List[Any] = [qvec, 'x', FILENAME_FILTER, qvec, limit_n]
                else:
                    # keep parameter count stable with the WHERE (%s IS NOT NULL)
                    params_vec = [qvec, None, qvec, limit_n]
                cur.execute(sql_vec, params_vec)
                vrows = cur.fetchall()
            except Exception:
                vrows = []

        # If lexical returned nothing at all, use wide OR ILIKE (not AND!)
        if not rows:
            terms = _informative_terms(query, max_terms=8)
            if terms:
                ors = " OR ".join([
                    f"{TXT_COL} ILIKE %s OR {META_COL}->>'page_title' ILIKE %s OR {META_COL}->>'section_path' ILIKE %s"
                    for _ in terms
                ])
                sql_like = f"""
                  SELECT
                    {ID_COL} AS id, {TXT_COL} AS document, {META_COL} AS metadata,
                    0.05::float AS ts_score, 0.0::float AS vec_score
                  FROM {TABLE}
                  WHERE {ors}{filter_clause}
                  LIMIT %s
                """
                params: List[Any] = []
                for t in terms:
                    pat = f"%{t}%"
                    params.extend([pat, pat, pat])
                params += filter_params + [limit_n]
                cur.execute(sql_like, params)
                rows = cur.fetchall()

        # Merge (sum scores) if vector was used
        if vrows:
            by_id: Dict[Any, Dict[str, Any]] = {}
            for r in rows + vrows:
                rid = r["id"]
                if rid not in by_id:
                    by_id[rid] = dict(r)
                else:
                    by_id[rid]["ts_score"]  = by_id[rid].get("ts_score", 0.0)  + float(r.get("ts_score") or 0.0)
                    by_id[rid]["vec_score"] = by_id[rid].get("vec_score", 0.0) + float(r.get("vec_score") or 0.0)
            rows = list(by_id.values())

    return rows

def _blend_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows: return rows
    ts = np.array([float(r.get("ts_score") or 0.0) for r in rows], dtype=float)
    vs = np.array([float(r.get("vec_score") or 0.0) for r in rows], dtype=float)

    def norm(a):
        if a.size==0: return a
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi <= lo + 1e-9: return np.zeros_like(a)
        return (a - lo) / max(1e-9, (hi-lo))

    tsn, vsn = norm(ts), norm(vs)
    alpha = HYBRID_ALPHA
    blend = alpha * vsn + (1.0 - alpha) * tsn

    out = []
    for i, r in enumerate(rows):
        rr = dict(r); rr["score"] = float(blend[i])
        out.append(rr)
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out

# ---------------------------
# Public API
# ---------------------------
def hybrid_retrieve_pg(query: str, top_k: int = 20, mmr_lambda: float = MMR_LAMBDA) -> List[Tuple[str, Dict[str, Any]]]:
    conn = _pg_connect()
    if conn is None: return []
    try:
        need = min(max(top_k * CAND_MULT, 50), MAX_SQL_LIMIT)
        base = _hybrid_candidates(conn, query, need)
    except Exception:
        base = []
    finally:
        try: conn.close()
        except Exception: pass

    if not base: return []

    blended = _blend_scores(base)

    # Prepare for MMR (use bag-of-words vecs as cheap proxies)
    cands: List[Dict[str, Any]] = []
    for r in blended:
        text = r.get("document") or ""
        meta_raw = r.get("metadata")
        source_val: Optional[str] = None
        if isinstance(meta_raw, dict):
            source_val = meta_raw.get("url") or meta_raw.get("source") or meta_raw.get("path") or None
        else:
            try:
                m = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
                source_val = m.get("url") or m.get("source") or m.get("path") or None
            except Exception:
                pass
        meta = {
            "id": str(r.get("id")) if r.get("id") is not None else None,
            "source": source_val,
            "score": float(r.get("score") or 0.0),
            "tfidf": _tf_dict(text),
        }
        cands.append({"text": text, "score": meta["score"], "tfidf": meta["tfidf"], "meta": meta})

    # Re-rank for diversity
    q_vec = _tf_dict(query)
    selected = mmr_select(cands, q_vec=q_vec, k=max(top_k, 5), lambda_=mmr_lambda)

    out: List[Tuple[str, Dict[str, Any]]] = []
    for s in selected[:top_k]:
        meta = dict(s["meta"]); meta.setdefault("id", None); meta.setdefault("source", None)
        out.append((s["text"], meta))
    return out

def format_result(row):
    """
    Normalize DB row into a canonical result dict.
    Prefer URL as the stable doc_id for alignment with eval/testset.
    """
    meta = row["metadata"]
    url = (meta.get("url") or "").strip().lower().rstrip("/") if isinstance(meta, dict) else ""

    return {
        "doc_id": url if url else row["id"],   # ✅ use URL if available
        "chunk_id": row["id"],                 # keep raw chunk ID for debugging
        "title": row.get("document", ""),      # chunk text
        "score": row.get("score", 0.0),
        "url": url,
    }

