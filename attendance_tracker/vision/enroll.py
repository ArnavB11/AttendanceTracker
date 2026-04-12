import cv2
from pathlib import Path
from enhancer import enhance_for_recognition
from recognizer import app, get_embedding
import pickle

demo_dir = Path("../data/demo_images")
embed_dir = Path("../data/face_embeddings")
embed_dir.mkdir(parents=True, exist_ok=True)

print("🔄 Starting Enrollment...\n")

count = 0
for img_path in sorted(demo_dir.glob("*.jpg")):
    print(f"Processing: {img_path.name}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"   ❌ Could not read image: {img_path.name}")
        continue

    # Resize image if it's too small (helps face detection)
    if img.shape[1] < 800:
        scale = 800 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    enhanced = enhance_for_recognition(img)
    emb = get_embedding(enhanced)

    if emb is not None:
        with open(embed_dir / f"{img_path.stem}.pkl", "wb") as f:
            pickle.dump(emb, f)
        print(f"   ✅ Enrolled successfully: {img_path.stem}")
        count += 1
    else:
        print(f"   ❌ No face detected in: {img_path.name}")
        print("      → Try using a clearer, closer, frontal face photo")

print(f"\n🎉 Enrollment completed! {count} student(s) enrolled.")