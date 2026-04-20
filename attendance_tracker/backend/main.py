import os
import sys
import csv
import base64
import numpy as np
import cv2
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add vision path to access our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vision')))

from enhancer import enhance_for_recognition
from recognizer import recognize_all

app = FastAPI(title="Attendance Tracker API")

# Setup CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ATTENDANCE_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'attendance.csv'))
STUDENTS_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'students.csv'))

# Create Attendance CSV if it doesn't exist
if not os.path.exists(ATTENDANCE_CSV):
    os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)
    with open(ATTENDANCE_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "RollNumber", "Date", "Time", "Status"])

# Global Cooldown Dictionary
daily_attendance = {}
active_session_students = []

class ScanRequest(BaseModel):
    image: str

@app.get("/api/attendance/live")
async def get_live_attendance():
    """Returns the list of highly-processed students recognized this session."""
    return {"status": "success", "identified_students": active_session_students}

@app.post("/api/scan")
async def scan_faces(req: ScanRequest):
    try:
        # Decode base64 image
        if "," in req.image:
            base64_data = req.image.split(",")[1]
        else:
            base64_data = req.image
            
        img_data = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
            
        # Run Vision Pipeline
        sharpened = enhance_for_recognition(frame)
        if sharpened is None:
            return {"status": "error", "message": "Enhancement failed."}
            
        results = recognize_all(sharpened)
        identified_students = []
        
        for admission_num_detected, similarity, bbox in results:
            if admission_num_detected and admission_num_detected != "Unknown":
                actual_admission_num = admission_num_detected.split("_")[-1] if "_" in admission_num_detected else admission_num_detected
                
                # DB lookup
                name_disp = "Unknown"
                class_disp = "Unknown"
                roll_disp = actual_admission_num
                
                try:
                    with open(STUDENTS_CSV, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row["Admission_Number"] == actual_admission_num or actual_admission_num in row["Full_Name"].lower():
                                name_disp = row["Full_Name"]
                                class_disp = row["Class_Section"]
                                roll_disp = row["Admission_Number"]
                                break
                except FileNotFoundError:
                    name_disp = admission_num_detected

                if name_disp == "Unknown":
                    name_disp = admission_num_detected.split("_")[0]

                # Cooldown check
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H:%M:%S")
                
                is_already_marked = (daily_attendance.get(actual_admission_num) == current_date)
                
                if not is_already_marked:
                    daily_attendance[actual_admission_num] = current_date
                    
                    # Log to CSV
                    with open(ATTENDANCE_CSV, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name_disp, roll_disp, current_date, current_time, "Present"])
                
                # Push to frontend regardless if they are marked today or not (so frontend sees them, frontend can handle filtering)
                # Or wait, the prompt says "just show name and attendance marked no other print statments . and also not repeat the names"
                # If we only return them when they are newly marked:
                if not is_already_marked:
                    student_data = {
                        "name": name_disp,
                        "roll": roll_disp,
                        "class": class_disp
                    }
                    identified_students.append(student_data)
                    active_session_students.append(student_data)

        return {"status": "success", "identified_students": identified_students}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
