import cv2
import time
import os
import csv
from datetime import datetime
from enhancer import enhance_for_recognition
from recognizer import recognize_all

print("🚀 Smart Attendance System Started")
print("Live feed is NORMAL. Checking every 5 seconds automatically.\n")

ATTENDANCE_CSV = "../data/attendance.csv"
PROOF_DIR = "../data/proof"

if not os.path.exists(ATTENDANCE_CSV):
    os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)
    with open(ATTENDANCE_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "RollNumber", "Date", "Time", "Status"])

if not os.path.exists(PROOF_DIR):
    os.makedirs(PROOF_DIR, exist_ok=True)

# Daily cooldown dictionary
# Format: {"John_101": "YYYY-MM-DD"}
daily_attendance = {}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(2)   # Camera warmup

last_check_time = time.time()
CHECK_INTERVAL_SECONDS = 5 # Reduced to 5 sec for more interactivity

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to read frame. Retrying...")
        time.sleep(1)
        continue

    # Show clean live feed
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Live Feed  |  Auto-check every {CHECK_INTERVAL_SECONDS}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Automatic check
    if time.time() - last_check_time >= CHECK_INTERVAL_SECONDS:
        last_check_time = time.time()
        print("\n📸 Taking snapshot for analysis...")

        sharpened = enhance_for_recognition(frame)

        if sharpened is not None:
            results = recognize_all(sharpened)
            
            # Make a copy of sharpened for drawing bounding boxes so the proof stays clean (or drawn on top of, your choice)
            drawn_sharpened = sharpened.copy()
            
            # Save proof if faces are processed
            if len(results) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Resize original to match sharpened if zoomed
                h_s, w_s = sharpened.shape[:2]
                frame_resized = cv2.resize(frame, (w_s, h_s))
                
                # Add text to them
                cv2.putText(frame_resized, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(sharpened, "Enhanced & Zoomed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Concatenate side by side
                proof_img = cv2.hconcat([frame_resized, sharpened])
                
                proof_path = os.path.join(PROOF_DIR, f"proof_{timestamp}.jpg")
                cv2.imwrite(proof_path, proof_img)
                print(f"📸 Proof saved to {proof_path}")
                
                # Show proof window briefly
                cv2.imshow("Enhancement Proof", proof_img)
            
            for index, (admission_num_detected, similarity, bbox) in enumerate(results):
                if admission_num_detected and admission_num_detected != "Unknown":
                    
                    actual_admission_num = admission_num_detected.split("_")[-1] if "_" in admission_num_detected else admission_num_detected
                    
                    # DB lookup 
                    name_disp = "Unknown"
                    class_disp = "Unknown"
                    roll_disp = actual_admission_num
                    
                    try:
                        with open("../data/students.csv", "r") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if row["Admission_Number"] == actual_admission_num or actual_admission_num in row["Full_Name"].lower():
                                    name_disp = row["Full_Name"]
                                    class_disp = row["Class_Section"]
                                    roll_disp = row["Admission_Number"]
                                    break
                    except FileNotFoundError:
                        # Fallback if DB doesn't exist yet
                        name_disp = admission_num_detected

                    # If name_disp is still unknown, it means the database lookup failed but they are enrolled.
                    if name_disp == "Unknown":
                        name_disp = admission_num_detected.split("_")[0]

                    # Cooldown check
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    is_already_marked = (daily_attendance.get(admission_num_detected) == current_date)
                    
                    if not is_already_marked:
                        daily_attendance[admission_num_detected] = current_date
                        print(f"✅ Student identified: {name_disp} (Roll: {roll_disp})")
                        print("✅ Attendance marked")
                        
                        # Log to CSV
                        with open(ATTENDANCE_CSV, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([name_disp, roll_disp, current_date, current_time, "Present"])
                    else:
                        print(f"ℹ️ {name_disp} (Roll: {roll_disp}) already marked present today.")
                    
                    # Draw Bounding Box and Text on the ENHANCED frame
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(drawn_sharpened, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{name_disp} | {roll_disp} | Present"
                    cv2.putText(drawn_sharpened, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(drawn_sharpened, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(drawn_sharpened, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if len(results) == 0:
                print("❌ No faces detected in snapshot")
                
            # Show a freeze frame for a bit to display results. We use drawn_sharpened so user sees the enhancement and zoom.
            cv2.imshow("Smart Attendance - Live Feed", drawn_sharpened)
            cv2.waitKey(2000) # Freeze for 2 seconds to show boxes
            
            # Check if Enhancement window needs to close
            try:
                if cv2.getWindowProperty("Enhancement Proof", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Enhancement Proof")
            except:
                pass
                
            continue # skip the normal imshow below for this iteration
        else:
            print("❌ Failed to enhance image")

    cv2.imshow("Smart Attendance - Live Feed", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped.")
