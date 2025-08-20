import os
import cv2
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity  
import logging
import time
from db_utils import load_embeddings_pg

# ----------------- Config -----------------
LOG_FILE = "verification_logs.log"
HIGH_THRESHOLD = 0.70
LOW_THRESHOLD  = 0.50

# ----------------- Logging -----------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------- Face Analysis -----------------
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def get_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding  


# ----------------- Verification -----------------
def verify_employee(empid, input_image, embeddings, low_thr=LOW_THRESHOLD, high_thr=HIGH_THRESHOLD):
    if embeddings is None or len(embeddings) == 0:
        msg = f"Employee '{empid}' not found in database"
        logging.warning(msg)
        return None, msg, None


    start_time = time.time()
    logging.info(f"Starting verification for {empid}")

    input_emb = get_embedding(input_image)
    if input_emb is None:
        msg = "Face not detected in input image"
        logging.warning(f"{empid} - {msg}")
        return None, msg, None

   
    best_sim = -1.0
    best_file = None
    status = "fail"
    for i in range(len(embeddings)):
        emb = np.array(embeddings[i]).reshape(1,-1)
        sim = cosine_similarity(emb, [input_emb])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_file = i+1
        if sim >= high_thr:
            status = "success"
            best_file = i+1
            break

    if status != "success":
        if best_sim >= high_thr:
            status = "success"
        elif best_sim >= low_thr:
            status = "partial"
        else:
            status = "fail"

    total_time = time.time() - start_time
    logging.info(
        f"{empid} - Verification {status.upper()} in {total_time:.3f} sec | "
        f"Best Similarity: {best_sim:.4f} | Image: {best_file}"
    )
    return status, best_sim, best_file

# ----------------- UI -----------------
st.title("Employee Face Verification System (PostgreSQL)")
st.write("Compare an uploaded or live image against employee embeddings stored in PostgreSQL.")

col_a, col_b = st.columns([1,1])
with col_a:
    ht = st.slider("High threshold (success)", 0.50, 0.95, HIGH_THRESHOLD, 0.01)
with col_b:
    lt = st.slider("Low threshold (partial lower bound)", 0.30, 0.90, LOW_THRESHOLD, 0.01)

empid = st.text_input("Enter Employee ID:").strip()
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Webcam"])

precomputed_embeddings = load_embeddings_pg(empid)

def render_result(status, best_sim, best_file):
    if status is None:
        st.error(best_sim)  # here best_sim is the error message
        return
    if status == "success":
        st.success(f"✅ Verification PASSED | Best Similarity: {best_sim:.4f} (matched: {best_file})")
    elif status == "partial":
        st.warning(f"🟨 Partial match | Best Similarity: {best_sim:.4f} (matched: {best_file})\n\n"
                   f"Proceed to Step-2 verification ")
    else:
        st.error("❌ Verification FAILED ")
# ---------- Tab 1: Upload ----------
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and empid:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        status, best_sim, best_file = verify_employee(empid, img, precomputed_embeddings, lt, ht)
        render_result(status, best_sim, best_file)

# ---------- Tab 2: Live Webcam ----------
with tab2:
    st.write("Capture from webcam for verification")
    live_frame = st.camera_input("Capture Face")
    if live_frame and empid:
        file_bytes = np.asarray(bytearray(live_frame.read()), dtype=np.uint8)
        live_img = cv2.imdecode(file_bytes, 1)
        #x = verify_employee(empid, live_img, precomputed_embeddings, lt, ht)
        
        status, best_sim, best_file = verify_employee(empid, live_img, precomputed_embeddings, lt, ht)
        render_result(status, best_sim, best_file)

