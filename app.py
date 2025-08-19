import os
import cv2 
import numpy as np 
import streamlit as st 
from insightface.app import FaceAnalysis 
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time

# ----------------- Logging Setup -----------------
LOG_FILE = "verification_logs.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------- Face Analysis Setup -----------------
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

EMPLOYEE_DIR = "employees"

def load_employee_images(empid):
    """Load all stored images for a given employee ID."""
    emp_dir = os.path.join(EMPLOYEE_DIR, empid)
    logging.info(f"Looking for employee folder: {emp_dir}")
    if not os.path.exists(emp_dir):
        logging.error(f"Folder not found: {emp_dir}")
        return []

    images = []
    for file in os.listdir(emp_dir):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(emp_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((file, img))   # store filename + image
            else:
                logging.warning(f"Failed to load image: {img_path}")
    return images

def get_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def verify_employee(empid, input_image, threshold=0.65):
    """Verify input image against multiple stored images of empid, returning all similarities."""
    start_time = time.time()
    logging.info(f"Starting verification for {empid}")

    # First, check input face
    input_emb = get_embedding(input_image)
    if input_emb is None:
        logging.warning(f"{empid} - No face detected in input image")
        return None, "Face not detected in input image", []

    # Load stored images
    stored_images = load_employee_images(empid)
    if not stored_images:
        logging.error(f"{empid} - No images found in database")
        return None, "Employee ID not found in database", []

    similarities = []
    verified = False
    best_sim = -1

    # Compute embeddings one by one
    for idx, (filename, img) in enumerate(stored_images):
        emb = get_embedding(img)
        if emb is None:
            logging.warning(f"{empid} - Face not detected in stored image {filename}")
            continue

        sim = cosine_similarity([emb], [input_emb])[0][0]
        similarities.append((filename, sim))
        logging.info(f"{empid} - Compared with {filename} | Similarity: {sim:.4f}")

        if sim > best_sim:
            best_sim = sim
        if sim > threshold:
            verified = True

    total_time = time.time() - start_time
    if verified:
        logging.info(f"{empid} - Verification PASSED in {total_time:.3f} sec | Best Similarity: {best_sim:.4f}")
    else:
        logging.info(f"{empid} - Verification FAILED in {total_time:.3f} sec | Best Similarity: {best_sim:.4f}")

    return verified, best_sim, similarities


# ----------------- Streamlit UI -----------------
st.title("Employee Face Verification System")
st.write("Verify uploaded image OR live webcam feed against employee database")

empid = st.text_input("Enter Employee ID:").strip()  # <-- strip spaces

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Webcam"])

# ---------- Tab 1: Upload ----------
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and empid:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result, best_sim, similarities = verify_employee(empid, img)

        if result is None:
            st.error(best_sim)  # here best_sim holds error message
        else:
            if result:
                st.success(f"✅ Verification Passed (Best Similarity: {best_sim:.4f})")
            else:
                st.error(f"❌ Verification Failed (Best Similarity: {best_sim:.4f})")

            # Show all similarity scores
            st.subheader("Similarity scores with stored images:")
            for fname, sim in similarities:
                st.write(f"**{fname}** → {sim:.4f}")

# ---------- Tab 2: Live Webcam ----------
with tab2:
    st.write("Click below to capture from webcam and verify:")

    # Live webcam input
    live_frame = st.camera_input("Capture Face")

    if live_frame and empid:
        file_bytes = np.asarray(bytearray(live_frame.read()), dtype=np.uint8)
        live_img = cv2.imdecode(file_bytes, 1)

        result, best_sim, similarities = verify_employee(empid, live_img)

        if result is None:
            st.error(best_sim)
        else:
            if result:
                st.success(f"✅ Verification Passed (Best Similarity: {best_sim:.4f})")
            else:
                st.error(f"❌ Verification Failed (Best Similarity: {best_sim:.4f})")

            st.subheader("Similarity scores with stored images:")
            for fname, sim in similarities:
                st.write(f"**{fname}** → {sim:.4f}")
