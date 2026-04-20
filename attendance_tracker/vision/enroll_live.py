import cv2
import os
import csv
import pickle
from datetime import datetime
from pathlib import Path
from enhancer import enhance_for_recognition
from recognizer import get_embedding

print("🚀 Starting Live Webcam Enrollment Tool")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Look at the camera and press SPACEBAR to capture your face.")
print("Press 'q' or 'ESC' to quit.")

captured_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    display = frame.copy()
    cv2.putText(display, "Press SPACE to Capture", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Live Enrollment", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32: # SPACE
        captured_frame = frame.copy()
        print("\n📸 SNAP! Face Captured.")
        break
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

if captured_frame is not None:
    print("\n--- Student Details ---")
    full_name = input("Enter Full Name: ").strip()
    admission_no = input("Enter Admission Number (e.g. 101): ").strip()
    class_sec = input("Enter Class Section (e.g. Sec-A): ").strip()
    
    if not full_name or not admission_no:
        print("❌ Enrollment Failed. Missing Name or Admission Number.")
    else:
        file_identifier = f"{full_name}_{admission_no}"
        
        # Save raw image
        demo_dir = Path("../data/demo_images")
        demo_dir.mkdir(parents=True, exist_ok=True)
        img_path = demo_dir / f"{file_identifier}.jpg"
        cv2.imwrite(str(img_path), captured_frame)
        
        # Process and save embedding
        print("⚙️ Processing Face Embedding...")
        enhanced = enhance_for_recognition(captured_frame)
        emb = get_embedding(enhanced)
        
        if emb is not None:
            embed_dir = Path("../data/face_embeddings")
            embed_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = embed_dir / f"{file_identifier}.pkl"
            
            with open(pkl_path, "wb") as f:
                pickle.dump(emb, f)
            print(f"✅ Embedding Saved at {pkl_path}")
            
            # Save mapping to DataBase CSV
            csv_path = "../data/students.csv"
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Admission_Number", "Full_Name", "Class_Section", "Registration_Date"])
                
                registration_date = datetime.now().strftime("%Y-%m-%d")
                writer.writerow([admission_no, full_name, class_sec, registration_date])
            
            print(f"🎉 Enrollment successful! Welcome {full_name}.")
        else:
            print("❌ Enrollment Failed! No face detected. Try again.")
else:
    print("Enrollment cancelled.")
