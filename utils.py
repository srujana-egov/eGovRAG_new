import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
import json
import tiktoken
from typing import Optional, Callable, Any

load_dotenv()

def get_env_var(name, default=None):
    return os.environ.get(name, default)

# Shared DB connection

def get_conn():
    conn = psycopg2.connect(
        dbname=get_env_var("PGDATABASE"),
        user=get_env_var("PGUSER"),
        password=get_env_var("PGPASSWORD"),
        host=get_env_var("PGHOST"),
        port=get_env_var("PGPORT"),
        sslmode=get_env_var("PGSSLMODE", "require")
    )
    register_vector(conn)
    return conn

# Shared chunk insert

def insert_chunk(doc_id, text, metadata, get_embedding):
    emb = get_embedding(text)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO hcmbot_knowledge (id, document, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (doc_id, text, emb, json.dumps(metadata)))
        conn.commit()

# Shared token counting

def count_tokens(text, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Check if chunk exists in DB

def chunk_exists(doc_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM hcmbot_knowledge WHERE id=%s LIMIT 1", (doc_id,))
            return cur.fetchone() is not None

def get_required_env(name: str, cast: Optional[Callable[[str], Any]] = None) -> Any:
    """
    Read environment variable `name`. Raise a RuntimeError if missing or empty.
    Optionally cast the string value using `cast` (e.g., int, bool).
    """
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Required environment variable {name!r} is not set or is empty.")
    if cast is not None:
        try:
            return cast(v)
        except Exception as e:
            raise RuntimeError(f"Error casting env var {name!r} value {v!r}: {e}")
    return v

