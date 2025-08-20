# offline_precompute_pg_images.py
import os
import cv2
import psycopg2
from insightface.app import FaceAnalysis

# ---------------- Config ----------------
EMPLOYEE_DIR = "employees"

DB_CONFIG = {
    "host": "localhost",   # change to your DB host
    "port": "5432",
    "dbname": "embeddings_db",
    "user": "postgres",
    "password": "abc24"
}

# ---------------- Init FaceAnalysis ----------------
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def get_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding  # length=512

# ---------------- DB Insert ----------------
def insert_employee(emp_id, embeddings):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    values = [emp_id]
    for i in range(4):
        if i < len(embeddings):
            values.append(embeddings[i].tolist())
        else:
            values.append(None)

    cur.execute("""
        INSERT INTO public.employee_emb ("emp_id", "EMB1", "EMB2", "EMB3", "EMB4")
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT ("emp_id") DO NOTHING;
    """, values)

    conn.commit()
    cur.close()
    conn.close()


# ---------------- Main ----------------
def main():
    for empid in os.listdir(EMPLOYEE_DIR):
        emp_dir = os.path.join(EMPLOYEE_DIR, empid)
        if not os.path.isdir(emp_dir):
            continue

        embeddings = []
        for file in sorted(os.listdir(emp_dir)):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                img_path = os.path.join(emp_dir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                emb = get_embedding(img)
                if emb is not None:
                    embeddings.append(emb)
        if embeddings:
            insert_employee(empid, embeddings)
            print(f"Inserted {len(embeddings)} embeddings for {empid}")

if __name__ == "__main__":
    main()
