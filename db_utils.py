import psycopg2
import logging
import numpy as np
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "embeddings_db",
    "user": "postgres",
    "password": "abc24"
}

# ----------------- Load embeddings from PostgreSQL -----------------
def load_embeddings_pg(emp_id):
    """Load embeddings from PostgreSQL and return a list of numpy arrays."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT "EMB1", "EMB2", "EMB3", "EMB4"
            FROM public.employee_emb
            WHERE emp_id = %s
        """, (emp_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is None:
            return []  # No embeddings found
        # Convert each stored embedding (assuming it's a list or array in DB) to np.array
        embeddings = [np.array(emb) for emb in row if emb is not None]
        return embeddings
    except Exception as e:
        logging.error(f"Error loading embeddings from PostgreSQL: {e}")
        return []

