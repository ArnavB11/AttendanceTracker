const video = document.getElementById("cameraFeed");
const scanBtn = document.getElementById("scanBtn");
const statusMsg = document.getElementById("statusMsg");
const scanOverlay = document.getElementById("scanOverlay");
const attendanceList = document.getElementById("attendanceList");
const emptyState = document.getElementById("emptyState");

let isScanning = false;
let scanIntervalId = null;
const markedStudents = new Set(); // To prevent duplicates in UI

// Create an offscreen canvas to capture images from the video stream
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

// Load any previous session data if the page is refreshed
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch("http://127.0.0.1:8000/api/attendance/live");
        const data = await response.json();
        if (data.status === "success" && data.identified_students) {
            data.identified_students.forEach(student => {
                if (!markedStudents.has(student.roll)) {
                    markedStudents.add(student.roll);
                    appendStudentToUI(student);
                }
            });
        }
    } catch (e) {
        console.warn("No active session data found or backend is offline.");
    }
});

scanBtn.addEventListener("click", async () => {
    if (!isScanning) {
        await startCamera();
    } else {
        stopCamera();
    }
});

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720 } 
        });
        
        video.srcObject = stream;
        
        // Wait for video to load metadata to get dimensions
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        isScanning = true;
        scanBtn.textContent = "Stop Scanner";
        scanBtn.classList.add("active");
        statusMsg.innerHTML = '<div class="status-dot live"></div> Scanner Live';
        scanOverlay.classList.remove("hidden");

        // Take snapshot and send to API every 5 seconds
        scanIntervalId = setInterval(captureAndScan, 5000);
        
    } catch (err) {
        console.error("Camera access denied or error: ", err);
        alert("Please grant camera permissions to use the scanner.");
    }
}

function stopCamera() {
    isScanning = false;
    scanBtn.textContent = "Initialize Scanner";
    scanBtn.classList.remove("active");
    statusMsg.innerHTML = '<div class="status-dot"></div> Camera Offline';
    scanOverlay.classList.add("hidden");

    clearInterval(scanIntervalId);

    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
}

async function captureAndScan() {
    if (!isScanning || !video.videoWidth) return;

    // Draw current video frame to offscreen canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64 jpeg
    const base64Image = canvas.toDataURL("image/jpeg", 0.9);

    try {
        const response = await fetch("http://127.0.0.1:8000/api/scan", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: base64Image })
        });

        const data = await response.json();
        
        if (data.status === "success" && data.identified_students.length > 0) {
            data.identified_students.forEach(student => {
                // Prevent duplicate UI entries during the session
                if (!markedStudents.has(student.roll)) {
                    markedStudents.add(student.roll);
                    appendStudentToUI(student);
                }
            });
        }
    } catch (error) {
        console.error("Backend API Error:", error);
    }
}

function appendStudentToUI(student) {
    if (emptyState) {
        emptyState.style.display = "none";
    }

    const card = document.createElement("div");
    card.className = "student-card";

    card.innerHTML = `
        <div class="student-info">
            <h3>${student.name}</h3>
            <p>Roll No: ${student.roll}</p>
        </div>
        <div class="status-badge">Present</div>
    `;

    // Add to the top of the list
    attendanceList.prepend(card);
}
