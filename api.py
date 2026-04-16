from fastapi import FastAPI, Query
from chunker import get_transformer_model
from typing import Optional
import psycopg2
import os

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
    return {"query": q, "filters": {"author": author, "after": after, "before": before}, "results": results}


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