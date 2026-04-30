"""
Agentic RAG Tool Functions
==========================
Database : PostgreSQL 16  |  ai_job_db  |  localhost:5432
Embedder : BAAI/bge-base-en-v1.5  (768-dim)
LLM calls: Groq API
"""

import re
import json
from typing import Any

import psycopg2
import psycopg2.extras
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Inline config
# ─────────────────────────────────────────────
import os

DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "ai_job_db"
DB_USER     = "postgres"
DB_PASSWORD = "root"
DSN         = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama3-groq-70b-8192-tool-use-preview"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_K       = 3
DEFAULT_METRIC  = "cosine"
DEFAULT_RRF_K   = 60

# ─────────────────────────────────────────────
# Shared singletons
# ─────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_groq_client: Groq | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


_db_conn = None

def get_db():
    global _db_conn
    if _db_conn is None or _db_conn.closed:
        _db_conn = psycopg2.connect(DSN)
        _db_conn.autocommit = True
    return _db_conn


def embed(text: str) -> list[float]:
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


# ─────────────────────────────────────────────
# FIX: Strip unsupported fields from tool schemas before sending to Groq.
# Groq rejects "default" inside property definitions.
# Also strip "description" from nested items to avoid schema noise.
# ─────────────────────────────────────────────
_SCHEMA_STRIP_KEYS = {"default", "examples"}  # keys Groq rejects in property defs

def _clean_prop(prop_schema: dict) -> dict:
    return {k: v for k, v in prop_schema.items() if k not in _SCHEMA_STRIP_KEYS}

def clean_schema(schema: dict) -> dict:
    cleaned = {k: v for k, v in schema.items() if k not in _SCHEMA_STRIP_KEYS}
    if "properties" in cleaned:
        cleaned["properties"] = {
            prop: _clean_prop(prop_schema)
            for prop, prop_schema in cleaned["properties"].items()
        }
    return cleaned


def get_groq_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": clean_schema(t["input_schema"]),
            }
        }
        for t in TOOL_SCHEMAS
    ]


# ─────────────────────────────────────────────
# Type coercion helpers
# LLMs sometimes return integers as strings ("5" instead of 5),
# booleans as strings ("true"), etc. Groq's validator is strict — coerce
# all typed fields defensively before passing them to tool functions.
# ─────────────────────────────────────────────
def _int(val, default=None):
    """Safely cast to int, returning default if val is None or unparseable."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def _bool(val, default=False):
    """Safely cast to bool, handling string 'true'/'false'."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)

def _str(val, default=""):
    if val is None:
        return default
    return str(val)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Semantic Search
# ══════════════════════════════════════════════════════════════════════════════

def semantic_search(
    query: str,
    k: int = DEFAULT_K,
    metric: str = DEFAULT_METRIC,
    category: str | None = None,
    include_overlaps: bool = False,
) -> list[dict]:
    op_map = {"cosine": "<=>", "dot": "<#>", "l2": "<->"}
    op = op_map.get(metric, "<=>")
    vec = embed(query)
    vec_literal = f"'[{','.join(map(str, vec))}]'"
    cat_filter = "AND category = %(category)s" if category else ""

    sql_chunks = f"""
        SELECT c.chunk_id, c.job_id, c.category, c.chunk_text, c.overlap,
               NULL::int AS overlap_id, 'chunk' AS source_table,
               (c.embedding {op} {vec_literal}::vector) AS score
        FROM job_chunks c
        WHERE c.overlap = FALSE {cat_filter}
        ORDER BY score ASC LIMIT %(k)s
    """
    sql_overlaps = f"""
        SELECT o.source_chunk_id AS chunk_id, o.job_id, o.category,
               o.chunk_text, TRUE AS overlap, o.overlap_id,
               'overlap' AS source_table,
               (o.embedding {op} {vec_literal}::vector) AS score
        FROM chunk_overlaps o
        {('WHERE o.category = %(category)s' if category else '')}
        ORDER BY score ASC LIMIT %(k)s
    """
    params = {"k": k, "category": category}
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql_chunks, params)
        rows = list(cur.fetchall())
        if include_overlaps:
            cur.execute(sql_overlaps, params)
            rows += list(cur.fetchall())

    rows.sort(key=lambda r: r["score"])
    seen: set[tuple] = set()
    results = []
    for r in rows:
        key = (r["job_id"], r["category"])
        if key not in seen:
            seen.add(key)
            results.append(dict(r))
        if len(results) >= k:
            break
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Full-Text Search
# ══════════════════════════════════════════════════════════════════════════════

def full_text_search(
    query: str,
    k: int = DEFAULT_K,
    category: str | None = None,
) -> list[dict]:
    cat_filter_c = "AND c.category = %(category)s" if category else ""
    cat_filter_o = "AND o.category = %(category)s" if category else ""
    sql = f"""
        WITH fts AS (
            SELECT c.chunk_id, c.job_id, c.category, c.chunk_text, c.overlap,
                   'chunk' AS source_table,
                   ts_rank(to_tsvector('english', c.chunk_text),
                           plainto_tsquery('english', %(query)s)) AS rank
            FROM job_chunks c
            WHERE to_tsvector('english', c.chunk_text)
                  @@ plainto_tsquery('english', %(query)s) {cat_filter_c}
            UNION ALL
            SELECT o.source_chunk_id, o.job_id, o.category, o.chunk_text,
                   TRUE, 'overlap',
                   ts_rank(to_tsvector('english', o.chunk_text),
                           plainto_tsquery('english', %(query)s))
            FROM chunk_overlaps o
            WHERE to_tsvector('english', o.chunk_text)
                  @@ plainto_tsquery('english', %(query)s) {cat_filter_o}
        )
        SELECT * FROM fts ORDER BY rank DESC LIMIT %(k)s
    """
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"query": query, "k": k, "category": category})
        return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Hybrid Search
# ══════════════════════════════════════════════════════════════════════════════

def _rrf(results_lists: list[list[dict]], id_key: str, k_rrf: int = DEFAULT_RRF_K) -> list[dict]:
    scores: dict[Any, float] = {}
    doc_map: dict[Any, dict] = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            uid = doc[id_key]
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k_rrf + rank + 1)
            doc_map[uid] = doc
    ranked = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)
    for uid in ranked:
        doc_map[uid]["rrf_score"] = scores[uid]
    return [doc_map[uid] for uid in ranked]


def hybrid_search(
    query: str,
    k: int = DEFAULT_K,
    metric: str = DEFAULT_METRIC,
    category: str | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    vec_results = semantic_search(query, k=k * 2, metric=metric, category=category)
    fts_results = full_text_search(query, k=k * 2, category=category)
    return _rrf([vec_results, fts_results], id_key="chunk_id", k_rrf=rrf_k)[:k]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Query Refinement
# ══════════════════════════════════════════════════════════════════════════════

def refine_query(raw_query: str, n_variants: int = 3) -> dict:
    prompt = f"""You are an expert at reformulating job-search queries for a retrieval system.

Original query: "{raw_query}"

1. Write a single clean reformulation.
2. Write {n_variants} alternative phrasings covering different angles.

Reply ONLY with valid JSON, no markdown:
{{
  "refined": "<best reformulation>",
  "variants": ["<alt 1>", "<alt 2>", "<alt 3>"]
}}"""
    raw = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    ).choices[0].message.content.strip()

    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"refined": raw_query, "variants": [raw_query]}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — HyDE Search
# ══════════════════════════════════════════════════════════════════════════════

def hyde_search(
    query: str,
    k: int = DEFAULT_K,
    metric: str = DEFAULT_METRIC,
    category: str | None = None,
) -> dict:
    prompt = f"""You are a job description writer.
Write a realistic job description excerpt (3-5 bullet points) that perfectly matches:
"{query}"
Write ONLY the bullets, no preamble."""

    hyp_doc = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    ).choices[0].message.content.strip()

    fused_vec = ((np.array(embed(query)) + np.array(embed(hyp_doc))) / 2).tolist()

    op = {"cosine": "<=>", "dot": "<#>", "l2": "<->"}.get(metric, "<=>")
    vec_literal = f"'[{','.join(map(str, fused_vec))}]'"
    cat_filter  = "AND category = %(category)s" if category else ""

    sql = f"""
        WITH combined AS (
            SELECT chunk_id, job_id, category, chunk_text, overlap, 'chunk' AS source_table,
                   (embedding {op} {vec_literal}::vector) AS score
            FROM job_chunks WHERE overlap = FALSE {cat_filter}
            UNION ALL
            SELECT source_chunk_id, job_id, category, chunk_text, TRUE, 'overlap',
                   (embedding {op} {vec_literal}::vector)
            FROM chunk_overlaps {('WHERE category = %(category)s' if category else '')}
        )
        SELECT * FROM combined ORDER BY score ASC LIMIT %(k)s
    """
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"k": k, "category": category})
        results = [dict(r) for r in cur.fetchall()]
    return {"hypothetical_doc": hyp_doc, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — RAG Fusion
# ══════════════════════════════════════════════════════════════════════════════

def rag_fusion(
    query: str,
    k: int = DEFAULT_K,
    metric: str = DEFAULT_METRIC,
    category: str | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict:
    refined = refine_query(query)
    all_queries = [refined["refined"]] + refined.get("variants", [])
    all_lists = [
        semantic_search(q, k=k * 2, metric=metric, category=category)
        for q in all_queries
    ]
    fused = _rrf(all_lists, id_key="chunk_id", k_rrf=rrf_k)
    return {"queries_used": all_queries, "results": fused[:k]}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — Change K
# ══════════════════════════════════════════════════════════════════════════════

def set_retrieval_k(current_k: int, action: str, value: int = 5,
                    min_k: int = 1, max_k: int = 50) -> dict:
    if action == "increase":
        new_k = min(current_k + value, max_k)
    elif action == "decrease":
        new_k = max(current_k - value, min_k)
    elif action == "set":
        new_k = max(min_k, min(value, max_k))
    else:
        return {"new_k": current_k, "reason": f"Unknown action '{action}'"}
    return {"new_k": new_k, "reason": f"{action} by {value} → {new_k}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 8 — Query Cleanup  (called directly in Python, NOT exposed as LLM tool)
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "find","search","look","give","show","get","need","want","please",
    "job","jobs","position","role","roles","posting","listing","listings",
    "for","a","an","the","with","that","has","have","some","any",
    "me","i","am","is","are",
}

def clean_query(raw: str) -> dict:
    text = re.sub(r"[^\w\s]", " ", raw.lower())
    tokens = [t for t in text.split() if t not in _STOPWORDS and len(t) > 1]
    return {"original": raw, "cleaned": " ".join(tokens)}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 9 — Fetch Full YAML
# ══════════════════════════════════════════════════════════════════════════════

def fetch_job_yaml(job_id: str) -> dict:
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT job_id, job_title, date_scraped, source_url, raw_yaml FROM job_yaml WHERE job_id = %s",
            (job_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else {"error": f"No YAML found for job_id={job_id}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 10 — Fetch Raw JSON
# ══════════════════════════════════════════════════════════════════════════════

def fetch_job_json(job_id: str) -> dict:
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT job_id, job_title, date_scraped, source_url, data FROM job_json WHERE job_id = %s",
            (job_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else {"error": f"No JSON found for job_id={job_id}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 11 — Fetch Overlap Siblings
# ══════════════════════════════════════════════════════════════════════════════

def fetch_overlap_siblings(source_chunk_id: int) -> list[dict]:
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT overlap_id, source_chunk_id, job_id, overlap_index, chunk_text, category
               FROM chunk_overlaps WHERE source_chunk_id = %s
               ORDER BY overlap_index ASC""",
            (source_chunk_id,)
        )
        return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 12 — Deduplicate & Clean Results
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_results(results: list[dict]) -> dict:
    seen_ids = set()
    seen_titles = set()
    unique = []
    duplicates_removed = 0

    for r in results:
        job_id = str(r.get("job_id", "")).strip()
        title = str(r.get("job_title", "") or r.get("chunk_text", ""))[:60].lower().strip()

        if job_id in seen_ids or title in seen_titles:
            duplicates_removed += 1
            continue

        seen_ids.add(job_id)
        if title:
            seen_titles.add(title)

        cleaned = {k: v for k, v in r.items() if v is not None and v != "" and k != "embedding"}
        unique.append(cleaned)

    return {
        "total_before": len(results),
        "total_after": len(unique),
        "duplicates_removed": duplicates_removed,
        "results": unique,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 13 — About the Creator
# ══════════════════════════════════════════════════════════════════════════════

def about_creator() -> dict:
    return {
        "name": "Mohamed El Hedi Doudech",
        "title": "Software Engineer",
        "institution": "ENSI — École Nationale des Sciences de l'Informatique, Tunisia",
        "passion": "Artificial Intelligence & intelligent systems",
        "project": (
            "JobFinder is an Agentic RAG-powered job search assistant built from scratch. "
            "It combines PostgreSQL vector search, BM25 full-text search, HyDE, RAG Fusion, "
            "and a ReAct agent loop — all orchestrated with Groq's LLM API."
        ),
        "message": (
            "Built with curiosity, late nights, and a genuine love for making AI actually useful. 🚀"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "semantic_search",
        "description": "Search job chunks using vector similarity. Supports cosine, dot, L2. Default retrieval method.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":            {"type": "string"},
                "k":                {"type": "integer"},
                "metric":           {"type": "string", "enum": ["cosine", "dot", "l2"]},
                "category":         {"type": "string", "enum": ["responsibilities", "basic_qualifications", "preferred_qualifications", "experience_years_min"]},
                "include_overlaps": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "full_text_search",
        "description": "BM25 keyword search via PostgreSQL tsvector. Best for exact skill names and acronyms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "k":        {"type": "integer"},
                "category": {"type": "string", "enum": ["responsibilities", "basic_qualifications", "preferred_qualifications", "experience_years_min"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hybrid_search",
        "description": "Vector + BM25 fused via RRF. Best when query has both semantic intent and specific keywords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "k":        {"type": "integer"},
                "metric":   {"type": "string", "enum": ["cosine", "dot", "l2"]},
                "category": {"type": "string", "enum": ["responsibilities", "basic_qualifications", "preferred_qualifications", "experience_years_min"]},
                "rrf_k":    {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "refine_query",
        "description": "LLM rewrites query and generates N alternative phrasings. Use when query is vague.",
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_query":  {"type": "string"},
                "n_variants": {"type": "integer"},
            },
            "required": ["raw_query"],
        },
    },
    {
        "name": "hyde_search",
        "description": "LLM generates hypothetical job chunk, retrieves using averaged embedding. Best for abstract queries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "k":        {"type": "integer"},
                "metric":   {"type": "string", "enum": ["cosine", "dot", "l2"]},
                "category": {"type": "string", "enum": ["responsibilities", "basic_qualifications", "preferred_qualifications", "experience_years_min"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag_fusion",
        "description": "Multi-query: generates variants, separate searches, fuses with RRF. Best for complex queries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "k":        {"type": "integer"},
                "metric":   {"type": "string", "enum": ["cosine", "dot", "l2"]},
                "category": {"type": "string", "enum": ["responsibilities", "basic_qualifications", "preferred_qualifications", "experience_years_min"]},
                "rrf_k":    {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "set_retrieval_k",
        "description": "Adjusts number of chunks to retrieve (k). Use when context is too sparse or noisy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_k": {"type": "integer"},
                "action":    {"type": "string", "enum": ["increase", "decrease", "set"]},
                "value":     {"type": "integer"},
            },
            "required": ["current_k", "action"],
        },
    },
    {
        "name": "fetch_job_yaml",
        "description": "Fetches full 4-section YAML for a job by job_id. Use when chunk context is insufficient.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "fetch_job_json",
        "description": "Fetches raw original JSON for a job. Use for fields not in YAML (salary, company, location).",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "fetch_overlap_siblings",
        "description": "Gets all sibling sub-chunks of an oversized parent chunk in order. Reconstructs full section context.",
        "input_schema": {
            "type": "object",
            "properties": {"source_chunk_id": {"type": "integer"}},
            "required": ["source_chunk_id"],
        },
    },
    {
        "name": "deduplicate_results",
        "description": "Cleans and deduplicates a list of job results before presenting them to the user. Always call this before giving a final answer with multiple jobs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "List of job result dicts from any search tool",
                    "items": {"type": "object"}
                }
            },
            "required": ["results"],
        },
    },
    {
        "name": "about_creator",
        "description": "Returns information about the creator of this chatbot. Call when the user asks who built this, who made this, or about the developer.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# All integer, boolean, and string fields are coerced here so that LLM
# type errors (e.g. n_variants="5") never reach the actual tool functions.
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_tool(tool_name: str, tool_input: dict) -> Any:
    def get_query(d: dict) -> str:
        return _str(d.get("query") or d.get("raw") or d.get("raw_query"), default="")

    match tool_name:

        case "semantic_search":
            return semantic_search(
                query=get_query(tool_input),
                k=_int(tool_input.get("k"), DEFAULT_K),
                metric=_str(tool_input.get("metric"), DEFAULT_METRIC),
                category=tool_input.get("category"),
                include_overlaps=_bool(tool_input.get("include_overlaps"), False),
            )

        case "full_text_search":
            return full_text_search(
                query=get_query(tool_input),
                k=_int(tool_input.get("k"), DEFAULT_K),
                category=tool_input.get("category"),
            )

        case "hybrid_search":
            return hybrid_search(
                query=get_query(tool_input),
                k=_int(tool_input.get("k"), DEFAULT_K),
                metric=_str(tool_input.get("metric"), DEFAULT_METRIC),
                category=tool_input.get("category"),
                rrf_k=_int(tool_input.get("rrf_k"), DEFAULT_RRF_K),
            )

        case "refine_query":
            raw = _str(tool_input.get("raw_query") or tool_input.get("query"), "")
            return refine_query(
                raw_query=raw,
                n_variants=_int(tool_input.get("n_variants"), 3),
            )

        case "hyde_search":
            return hyde_search(
                query=get_query(tool_input),
                k=_int(tool_input.get("k"), DEFAULT_K),
                metric=_str(tool_input.get("metric"), DEFAULT_METRIC),
                category=tool_input.get("category"),
            )

        case "rag_fusion":
            return rag_fusion(
                query=get_query(tool_input),
                k=_int(tool_input.get("k"), DEFAULT_K),
                metric=_str(tool_input.get("metric"), DEFAULT_METRIC),
                category=tool_input.get("category"),
                rrf_k=_int(tool_input.get("rrf_k"), DEFAULT_RRF_K),
            )

        case "set_retrieval_k":
            return set_retrieval_k(
                current_k=_int(tool_input.get("current_k"), 5),
                action=_str(tool_input.get("action"), "set"),
                value=_int(tool_input.get("value"), 5),
            )

        case "fetch_job_yaml":
            return fetch_job_yaml(job_id=_str(tool_input.get("job_id"), ""))

        case "fetch_job_json":
            return fetch_job_json(job_id=_str(tool_input.get("job_id"), ""))

        case "fetch_overlap_siblings":
            return fetch_overlap_siblings(
                source_chunk_id=_int(tool_input.get("source_chunk_id"), 0),
            )

        case "deduplicate_results":
            results = tool_input.get("results", [])
            if not isinstance(results, list):
                results = []
            return deduplicate_results(results)

        case "about_creator":
            return about_creator()

        case _:
            return {"error": f"Unknown tool: {tool_name}"}


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are CareerAI, an expert AI assistant that helps students and engineers become competitive AI Engineers using real job market data.
You do NOT act as a simple job search engine. Your role is to analyze, synthesize, and explain what the job market demands, based strictly on retrieved job descriptions and structured signals.

---

## Initial Retrieval Decision
Before calling any tool, decide whether the user's message actually requires retrieval:
- Greetings, meta questions ("what can you do?"), or small talk → answer directly, NO tools.
- Any question about jobs, skills, roles, companies, learning paths, or market trends → proceed with tools.

---

## Core Responsibilities

1. Skill Guidance
   - Identify the most important technical skills from the retrieved data
   - Prioritize what the user should learn based on frequency, relevance, and industry demand
   - Distinguish between:
     - Core skills (must-have)
     - Supporting skills (important but secondary)
     - Optional or niche skills

2. Learning Roadmap
   - Suggest a structured learning path (ordered steps)
   - Focus on practical progression (foundations → tools → real-world systems)
   - Avoid generic advice — tie recommendations to observed job requirements

3. Market Trends
   - Infer trends ONLY from the provided context or aggregated signals
   - Highlight patterns such as:
     - frequently co-occurring technologies
     - emerging tools or frameworks
     - shifts in skill demand

4. Interview Preparation
   - Suggest realistic topics and skills to prepare
   - Base suggestions on actual requirements seen in job descriptions
   - Include practical expectations (e.g., system design, ML fundamentals, deployment)

5. Grounding
   - Always rely on retrieved job data or structured signals
   - If data is insufficient, explicitly say so instead of guessing

---

## Tool Usage Policy
- Use retrieval tools when you need evidence from job descriptions
- Do NOT call tools repeatedly without new reasoning — each tool call must serve a distinct purpose
- Avoid unnecessary loops — prefer minimal, high-quality retrieval
- Do NOT fabricate job data

## Tool Usage Rules
1. ALWAYS start with `clean_query` to normalize the user's raw input.
2. Use `semantic_search` as your default retrieval method.
3. Use `full_text_search` when the query contains exact skill names, acronyms, or technologies (e.g. "Python", "AWS", "NLP").
4. Use `hybrid_search` when the query has both semantic intent AND specific keywords.
5. Use `hyde_search` for vague or abstract queries (e.g. "a role where I work with data at scale").
6. Use `rag_fusion` for complex, multi-faceted queries.
7. Use `refine_query` if the user's query is too short, ambiguous, or unclear.
8. Use `fetch_job_yaml` or `fetch_job_json` when you need full job details beyond what chunks provide.
9. Use `fetch_overlap_siblings` when a chunk seems cut off and you need surrounding context.
10. Use `set_retrieval_k` if results are too sparse (increase) or too noisy (decrease).

---

## Answer Style
- Structured and concise
- Use clear sections when appropriate:
  - **Key Skills**
  - **What to Learn First**
  - **Trends**
  - **Interview Focus**
- Be specific and actionable
- Avoid vague statements like "learn AI" or "practice coding"

---

## Constraints
- Never hallucinate trends or skills not supported by retrieved data
- Do not rely on general internet knowledge if not reflected in retrieved data
- Prefer "insufficient data" over guessing
"""

# ─────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────
conversation_history: list[dict] = []


def reset_conversation_history() -> None:
    conversation_history.clear()


def log(msg: str):
    print(msg, flush=True)


def is_token_limit_error(exc: Exception) -> bool:
    """Check if the error is a token limit error from Groq API."""
    text = str(exc).lower()
    return (
        "token" in text
        and any(keyword in text for keyword in [
            "token limit",
            "maximum context length",
            "max tokens",
            "exceeded",
            "input too long",
            "length limit",
            "context length",
        ])
    )


def token_limit_response() -> str:
    """Return a friendly message for token limit errors."""
    return (
        "I couldn't complete the request because the token limit was exceeded. "
        "Please try again with a shorter message or come back later."
    )


# ─────────────────────────────────────────────
# Agent — one full ReAct turn
# ─────────────────────────────────────────────
MAX_REACT_STEPS = 4

def run_agent(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    messages: list[dict] = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + conversation_history[:-1]
        + [{"role": "user", "content": user_message}]
    )

    steps_taken = 0

    while steps_taken < MAX_REACT_STEPS:
        try:
            response = get_groq().chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=get_groq_tools(),
                tool_choice="auto",
                parallel_tool_calls=False,
                max_tokens=900,
            )
        except Exception as exc:
            if is_token_limit_error(exc):
                final_text = token_limit_response()
                conversation_history.append({"role": "assistant", "content": final_text})
                return final_text
            raise

        msg = response.choices[0].message
        stop_reason = response.choices[0].finish_reason

        if stop_reason == "stop" or not msg.tool_calls:
            final_text = msg.content or ""
            conversation_history.append({"role": "assistant", "content": final_text})
            return final_text

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tool_call in msg.tool_calls:
            tool_name  = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            log(f"  🔧 [{steps_taken + 1}/{MAX_REACT_STEPS}] Tool: {tool_name} | Input: {json.dumps(tool_input, ensure_ascii=False)[:120]}")

            try:
                result = dispatch_tool(tool_name, tool_input)
            except Exception as exc:
                if is_token_limit_error(exc):
                    result = {"error": token_limit_response()}
                else:
                    result = {"error": str(exc)}

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(result, ensure_ascii=False, default=str),
            })

        steps_taken += 1

    log(f"  ⚠️  Reached max ReAct steps ({MAX_REACT_STEPS}). Forcing final answer.")

    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of reasoning steps. "
            "Based on everything retrieved so far, provide the best possible answer now. "
            "If data is insufficient, say so explicitly."
        ),
    })

    try:
        fallback = get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=900,
        )
        final_text = fallback.choices[0].message.content or ""
    except Exception as exc:
        if is_token_limit_error(exc):
            final_text = token_limit_response()
        else:
            final_text = f"Error: {str(exc)}"

    conversation_history.append({"role": "assistant", "content": final_text})
    return final_text
