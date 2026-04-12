import cv2
from pathlib import Path
from enhancer import enhance_for_recognition
from recognizer import app, get_embedding
import pickle

demo_dir = Path("../data/demo_images")
embed_dir = Path("../data/face_embeddings")
embed_dir.mkdir(parents=True, exist_ok=True)

print("🔄 Starting Enrollment with Diagnostics...\n")

count = 0
for img_path in sorted(demo_dir.glob("*.*")):   # Check all image types
    print(f"Processing: {img_path.name}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"   ❌ Could not read image: {img_path.name}")
        continue

    print(f"   Image size: {img.shape[1]}x{img.shape[0]}")

    # Resize for better detection
    if img.shape[1] < 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"   Resized to: {img.shape[1]}x{img.shape[0]}")

    enhanced = enhance_for_recognition(img)
    emb = get_embedding(enhanced)

    if emb is not None:
        with open(embed_dir / f"{img_path.stem}.pkl", "wb") as f:
            pickle.dump(emb, f)
        print(f"   ✅ Enrolled successfully: {img_path.stem}")
        count += 1
    else:
        print(f"   ❌ No face detected in: {img_path.name}")
        print("      → Tips: Use clearer, closer, frontal photo with good lighting")

print(f"\n🎉 Enrollment completed! {count} student(s) enrolled.")