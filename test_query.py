import psycopg2
from chunker import get_transformer_model

# --- CONFIG ---
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres"
}

# --- CONNECT DB ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# --- EMBEDDING FUNCTION ---
def embed(text):
    model = get_transformer_model()
    return model.encode(text).tolist()

# --- QUERY FUNCTION ---
def retrieve(query):
    q_emb = embed(query)

    cur.execute("""
        SELECT content, section, paper_id,
               embedding <-> %s::vector AS distance
        FROM chunks
        ORDER BY embedding <-> %s::vector
        LIMIT 5;
    """, (q_emb, q_emb))

    return cur.fetchall()

# --- RUN TEST ---
if __name__ == "__main__":
    query = "What is quantisation?"

    results = retrieve(query)

    for r in results:
        print("\n---")
        print("Section:", r[1])
        print("Distance:", r[3])
        print(r[0][:300])