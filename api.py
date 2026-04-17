from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from chunker import get_transformer_model
from generator import generate_cited_answer
from typing import Optional
import psycopg2
import os
import logging
import json
from datetime import datetime

# Set up logging
LOG_FILE = "logs/query_logs.jsonl"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG ArXiv Retrieval API")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5433)),
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def log_query(entry: dict):
    """Append a structured log entry to the JSONL file."""
    entry["timestamp"] = datetime.utcnow().isoformat()
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Query logged: %s q='%s'", entry["endpoint"], entry["query"])


def retrieve_chunks(query_embedding, top_k, author=None, after=None, before=None):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SET ivfflat.probes = 3")

    filters = []
    params = {"emb": query_embedding, "top_k": top_k}

    if author:
        filters.append("%(author)s = ANY(p.authors)")
        params["author"] = author
    if after:
        filters.append("p.published >= %(after)s::date")
        params["after"] = after
    if before:
        filters.append("p.published <= %(before)s::date")
        params["before"] = before

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    cur.execute(
        f"""
        SELECT c.section, c.content, 1 - (c.embedding <=> %(emb)s::vector) AS similarity,
               p.arxiv_id, p.title, p.authors, p.published
        FROM chunks c
        JOIN papers p ON c.paper_id = p.id
        {where_clause}
        ORDER BY c.embedding <=> %(emb)s::vector
        LIMIT %(top_k)s
        """,
        params,
    )
    results = [
        {
            "section": row[0],
            "content": row[1][:500],
            "similarity": round(float(row[2]), 4),
            "arxiv_id": row[3],
            "title": row[4],
            "authors": row[5],
            "published": str(row[6]) if row[6] else None,
        }
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return results


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = 5,
    author: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    model = get_transformer_model()
    query_embedding = model.encode(q).tolist()
    results = retrieve_chunks(query_embedding, top_k, author, after, before)

    log_query({
        "endpoint": "/search",
        "query": q,
        "filters": {"author": author, "after": after, "before": before},
        "top_k": top_k,
        "num_results": len(results),
        "top_similarities": [r["similarity"] for r in results],
        "retrieved_papers": [{"arxiv_id": r["arxiv_id"], "section": r["section"]} for r in results],
    })

    return {"query": q, "filters": {"author": author, "after": after, "before": before}, "results": results}


@app.get("/ask")
def ask(
    q: str = Query(..., description="Your question"),
    top_k: int = 5,
    author: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    model = get_transformer_model()
    query_embedding = model.encode(q).tolist()
    chunks = retrieve_chunks(query_embedding, top_k, author, after, before)

    generation = generate_cited_answer(q, chunks)

    references = [
        {
            "ref": f"[{i}]",
            "arxiv_id": c["arxiv_id"],
            "title": c["title"],
            "section": c["section"],
            "similarity": c["similarity"],
        }
        for i, c in enumerate(chunks, 1)
    ]

    # Build the prompt that was sent to the LLM (for debugging)
    context_sent = ""
    for i, chunk in enumerate(chunks, 1):
        context_sent += f"[{i}] (Paper: {chunk['arxiv_id']}, Section: {chunk['section']})\n"
        context_sent += f"{chunk['content']}\n\n"

    log_query({
        "endpoint": "/ask",
        "query": q,
        "filters": {"author": author, "after": after, "before": before},
        "top_k": top_k,
        "num_results": len(chunks),
        "top_similarities": [c["similarity"] for c in chunks],
        "retrieved_papers": [{"arxiv_id": c["arxiv_id"], "section": c["section"]} for c in chunks],
        "context_sent_to_llm": context_sent,
        "llm_answer": generation["answer"],
        "model": generation["model"],
        "tokens_used": generation["tokens_used"],
    })

    return {
        "query": q,
        "answer": generation["answer"],
        "references": references,
        "model": generation["model"],
        "tokens_used": generation["tokens_used"],
    }


@app.get("/logs")
def get_logs(n: int = 20):
    """View the last n query logs."""
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"total_logs": len(lines), "showing": len(entries), "logs": entries}
    except FileNotFoundError:
        return {"total_logs": 0, "showing": 0, "logs": []}


@app.get("/logs/view")
def logs_view():
    return FileResponse("static/logs.html")


@app.get("/papers")
def list_papers():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT arxiv_id, title, authors, published FROM papers ORDER BY published DESC")
    papers = [
        {
            "arxiv_id": row[0],
            "title": row[1],
            "authors": row[2],
            "published": str(row[3]) if row[3] else None,
        }
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return {"papers": papers}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")