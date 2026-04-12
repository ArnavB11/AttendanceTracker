import cv2
import time
from enhancer import enhance_for_recognition
from recognizer import recognize

print("🚀 Smart Attendance System Started")
print("Live feed is NORMAL. Checking every 15 seconds automatically.\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(2)   # Camera warmup

last_check_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to read frame. Retrying...")
        time.sleep(1)
        continue

    # Show clean live feed
    display_frame = frame.copy()
    cv2.putText(display_frame, "Live Feed  |  Auto-check every 15s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance - Live Feed", display_frame)

    # Automatic check every 15 seconds
    if time.time() - last_check_time >= 15:
        last_check_time = time.time()
        print("\n📸 Taking snapshot for analysis...")

        sharpened = enhance_for_recognition(frame)

        if sharpened is None:
            print("❌ Failed to enhance image")
            continue

        student_name, similarity = recognize(sharpened)

        if student_name:
            print(f"✅ Student identified: {student_name}")
            print("✅ Attendance marked")
        else:
            print("❌ No match found - Try again")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped.")