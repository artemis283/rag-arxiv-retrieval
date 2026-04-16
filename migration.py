import psycopg2
import os

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5433)),
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Add metadata columns to papers table
cur.execute("""
    ALTER TABLE papers ADD COLUMN IF NOT EXISTS authors TEXT[];
    ALTER TABLE papers ADD COLUMN IF NOT EXISTS published DATE;
    ALTER TABLE papers ADD COLUMN IF NOT EXISTS abstract TEXT;
""")

conn.commit()
cur.close()
conn.close()
print("Migration complete!")