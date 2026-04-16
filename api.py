from fastapi import FastAPI, Query
from chunker import get_transformer_model
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
def search(q: str = Query(..., description="Search query"), top_k: int = 5):
    model = get_transformer_model()
    query_embedding = model.encode(q).tolist()

    conn = get_connection()
    cur = conn.cursor()

    # Set number of clusters to probe 
    cur.execute("SET ivfflat.probes = 3")

    cur.execute(
        """
        SELECT c.section, c.content, 1 - (c.embedding <=> %s::vector) AS similarity, p.arxiv_id
        FROM chunks c
        JOIN papers p ON c.paper_id = p.id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )
    results = [
        {
            "section": row[0],
            "content": row[1][:500],
            "similarity": round(float(row[2]), 4),
            "arxiv_id": row[3],
        }
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return {"query": q, "results": results}


@app.get("/health")
def health():
    return {"status": "ok"}