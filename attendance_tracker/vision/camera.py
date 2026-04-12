import cv2
import time
from enhancer import enhance_for_recognition
from recognizer import recognize

print("🚀 Smart Attendance System Started")
print("Camera is NORMAL. Checking every 15 seconds...\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_check_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show completely normal live feed (no sharpening visible)
    display_frame = frame.copy()

    # Automatic check every 15 seconds
    if time.time() - last_check_time >= 15:
        last_check_time = time.time()
        
        print("\n📸 Taking snapshot for analysis...")

        # Sharpen ONLY the snapshot (not shown on screen)
        sharpened_snapshot = enhance_for_recognition(frame)

        # Analyze the sharpened image
        student_name, similarity = recognize(sharpened_snapshot)

        if student_name:
            print(f"✅ Student identified: {student_name}")
            print("✅ Attendance marked")
        else:
            print("❌ No match found - Try again")

    # Optional small text on screen
    cv2.putText(display_frame, "Live Feed  |  Auto-check every 15s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance - Live Feed", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()