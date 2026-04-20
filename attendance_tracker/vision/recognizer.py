import cv2
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

def load_embeddings():
    embeddings = {}
    embed_dir = Path("../data/face_embeddings")
    for pkl_file in embed_dir.glob("*.pkl"):
        student_name = pkl_file.stem
        with open(pkl_file, "rb") as f:
            embeddings[student_name] = pickle.load(f)
    return embeddings

known_embeddings = load_embeddings()

def get_embedding(frame: np.ndarray):
    """Improved face detection"""
    if frame is None:
        return None
    
    # Try original size
    faces = app.get(frame)
    if len(faces) > 0:
        return max(faces, key=lambda x: x.det_score).normed_embedding
    
    # Resize and try again (helps when face is small)
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    faces = app.get(resized)
    
    if len(faces) > 0:
        print("   (Face detected after resizing)")
        return max(faces, key=lambda x: x.det_score).normed_embedding
    
    return None

def recognize(frame: np.ndarray, threshold: float = 0.42):
    """Return student name and similarity"""
    embedding = get_embedding(frame)
    if embedding is None:
        return None, 0.0

    best_name = None
    best_sim = -1.0

    for name, known_emb in known_embeddings.items():
        sim = 1 - cosine(embedding, known_emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_sim > threshold:
        return best_name, best_sim
    return None, best_sim

def get_all_embeddings(frame: np.ndarray):
    """Returns a list of (embedding, bbox) for all detected faces"""
    if frame is None:
        return []
    
    faces = app.get(frame)
    if len(faces) > 0:
        return [(f.normed_embedding, f.bbox) for f in faces]
    
    # Try resizing
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    faces = app.get(resized)
    
    if len(faces) > 0:
        print("   (Faces detected after resizing)")
        results = []
        for f in faces:
            adjusted_bbox = [coord / 2.0 for coord in f.bbox]
            results.append((f.normed_embedding, adjusted_bbox))
        return results
    
    return []

def recognize_all(frame: np.ndarray, threshold: float = 0.42):
    """Return a list of (student_name, similarity, bbox)"""
    detections = get_all_embeddings(frame)
    results = []
    
    for embedding, bbox in detections:
        best_name = None
        best_sim = -1.0
        
        for name, known_emb in known_embeddings.items():
            sim = 1 - cosine(embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name
                
        if best_sim > threshold:
            results.append((best_name, best_sim, bbox))
        else:
            results.append(("Unknown", best_sim, bbox))
            
    return results