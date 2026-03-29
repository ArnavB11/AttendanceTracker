from insightface.app import FaceAnalysis
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cosine

# Initialize InsightFace once
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))   # Use CPU since no GPU mentioned

def load_embeddings():
    """Load all stored embeddings"""
    embeddings = {}
    embed_dir = Path("data/face_embeddings")
    if not embed_dir.exists():
        embed_dir.mkdir(parents=True, exist_ok=True)
        return embeddings
    
    for pkl_file in embed_dir.glob("*.pkl"):
        value = pkl_file.stem
        with open(pkl_file, "rb") as f:
            embeddings[value] = pickle.load(f)
    return embeddings

known_embeddings = load_embeddings()

def get_embedding(frame: np.ndarray):
    """Detect face and extract embedding"""
    if frame is None:
        return None
    faces = app.get(frame)
    if len(faces) == 0:
        return None
    # Take the best detected face
    face = max(faces, key=lambda x: x.det_score)
    return face.normed_embedding

def recognize(frame: np.ndarray, threshold: float = 0.45):
    """Return matched value or None"""
    embedding = get_embedding(frame)
    if embedding is None:
        return None, 0.0
    
    best_value = None
    best_sim = -1.0
    
    for value, known_emb in known_embeddings.items():
        sim = 1 - cosine(embedding, known_emb)
        if sim > best_sim:
            best_sim = sim
            best_value = value
    
    if best_sim > threshold:
        return best_value, best_sim
    return None, best_sim