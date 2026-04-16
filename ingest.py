import psycopg2
from pathlib import Path
from chunker import chunk_by_section, get_transformer_model
import os

LATEX_DIR = Path('fixtures/latex')

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 5433)),
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def ingest_paper(conn, paper_dir: Path):
    """Chunk a paper, embed it, and store in Postgres."""
    arxiv_id = paper_dir.name
    chunks = chunk_by_section(paper_dir)

    if not chunks:
        print(f"  Skipping {arxiv_id} — no chunks found")
        return

    model = get_transformer_model()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO papers (arxiv_id, title)
        VALUES (%s, %s)
        ON CONFLICT (arxiv_id) DO NOTHING
        RETURNING id
        """,
        (arxiv_id, chunks[0]['section']),
    )
    row = cur.fetchone()

    if row is None:
        print(f"  Skipping {arxiv_id} — already ingested")
        cur.close()
        return

    paper_id = row[0]

    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            """
            INSERT INTO chunks (paper_id, section, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (paper_id, chunk['section'], i, chunk['text'], embedding.tolist()),
        )

    conn.commit()
    cur.close()
    print(f"  Ingested {arxiv_id} — {len(chunks)} chunks")


def main():
    conn = get_connection()

    paper_dirs = sorted(p for p in LATEX_DIR.iterdir() if p.is_dir())
    print(f"Found {len(paper_dirs)} papers to ingest.\n")

    for paper_dir in paper_dirs:
        print(f"Processing {paper_dir.name}...")
        try:
            ingest_paper(conn, paper_dir)
        except Exception as e:
            print(f"  Error: {e}")
            conn.rollback()

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()