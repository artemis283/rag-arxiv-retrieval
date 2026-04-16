import psycopg2
from pathlib import Path
from chunker import chunk_by_section, get_transformer_model
from metadata_fetcher import fetch_metadata
import os
import time

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5433)),
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

model = get_transformer_model()

papers_dir = Path("fixtures/latex")

for paper_dir in sorted(papers_dir.iterdir()):
    if not paper_dir.is_dir():
        continue
    arxiv_id = paper_dir.name
    print(f"Processing {arxiv_id}...")

    chunks = chunk_by_section(paper_dir)
    if not chunks:
        print(f"  Skipping {arxiv_id} — no chunks found")
        continue

    try:
        # Fetch metadata from arXiv API
        meta = fetch_metadata(arxiv_id)
        title = meta.get("title", "Unknown")
        authors = meta.get("authors", [])
        published = meta.get("published", None)
        abstract = meta.get("abstract", None)
        time.sleep(0.5)  # Rate limit

        cur.execute(
            """
            INSERT INTO papers (arxiv_id, title, authors, published, abstract)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO UPDATE
            SET title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                published = EXCLUDED.published,
                abstract = EXCLUDED.abstract
            RETURNING id
            """,
            (arxiv_id, title, authors, published, abstract),
        )
        paper_id = cur.fetchone()[0]

        # Delete old chunks for this paper (in case of re-ingest)
        cur.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk["text"]).tolist()
            cur.execute(
                """
                INSERT INTO chunks (paper_id, section, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (paper_id, chunk["section"], i, chunk["text"], embedding),
            )

        conn.commit()
        print(f"  Ingested {arxiv_id} — {len(chunks)} chunks | {title[:60]}")

    except Exception as e:
        conn.rollback()
        print(f"  Error: {e}")

cur.close()
conn.close()
print("\nDone!")