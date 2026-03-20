import psycopg2
from pathlib import Path
from chunker import chunk_by_section, get_transformer_model

LATEX_DIR = Path('fixtures/latex/2603.01399v1')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',  # update if different
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

    # Insert paper record (skip if already exists)
    cur.execute(
        """
        INSERT INTO papers (arxiv_id, title)
        VALUES (%s, %s)
        ON CONFLICT (arxiv_id) DO NOTHING
        RETURNING id
        """,
        (arxiv_id, chunks[0]['section']),  # use first section as rough title
    )
    row = cur.fetchone()

    if row is None:
        print(f"  Skipping {arxiv_id} — already ingested")
        cur.close()
        return

    paper_id = row[0]

    # Embed and insert each chunk
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

    print(f"Processing {LATEX_DIR.name}...")
    try:
        ingest_paper(conn, LATEX_DIR)
    except Exception as e:
        print(f"  Error: {e}")
        conn.rollback()

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()