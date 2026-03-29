import cv2
from enhancer import enhance_for_recognition
from recognizer import recognize

print("📸 Starting Laptop-Only Attendance Demo")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Your team's enhancement
    enhanced = enhance_for_recognition(frame)
    
    # Recognition
    user_value, similarity = recognize(enhanced)
    
    if user_value:
        text = f"Matched: {user_value} ({similarity:.3f})"
        color = (0, 255, 0)
        print(f"✅ MATCH → {user_value} (similarity: {similarity:.3f})")
    else:
        text = "No match"
        color = (0, 0, 255)
    
    cv2.putText(enhanced, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, color, 3)
    
    cv2.imshow("Smart Attendance - Sharpened Demo", enhanced)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()