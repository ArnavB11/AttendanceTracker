import cv2
from pathlib import Path
from enhancer import enhance_for_recognition
from recognizer import app, get_embedding
import pickle

demo_dir = Path("data/demo_images")
embed_dir = Path("data/face_embeddings")
embed_dir.mkdir(parents=True, exist_ok=True)

print("🔄 Starting enrollment from data/demo_images/")

for img_path in demo_dir.glob("*.jpg"):
    print(f"Processing: {img_path.name}")
    img = cv2.imread(str(img_path))
    if img is None:
        print("   ❌ Failed to read image")
        continue
    
    enhanced = enhance_for_recognition(img)
    embedding = get_embedding(enhanced)
    
    if embedding is None:
        print("   ❌ No face detected")
        continue
    
    value = img_path.stem
    with open(embed_dir / f"{value}.pkl", "wb") as f:
        pickle.dump(embedding, f)
    
    print(f"   ✅ Enrolled: {value}")

print("\n🎉 Enrollment completed!")