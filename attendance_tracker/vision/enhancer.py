import cv2
import numpy as np

def auto_brighten(frame: np.ndarray) -> np.ndarray:
    """Brightens the image if it is too dark using CLAHE."""
    if frame is None:
        return None
        
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Calculate average brightness
    avg_brightness = np.mean(v)
    
    # If the image is dark, apply CLAHE on the V channel
    if avg_brightness < 100:  
        print("   (Auto-Brightening Applied)")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        
        # Merge back
        hsv_enhanced = cv2.merge((h, s, v_enhanced))
        frame = cv2.cvtColor(hsv_enhanced, cv2.HSV2BGR)

    return frame

def auto_zoom(frame: np.ndarray) -> np.ndarray:
    """Detects faces using a fast Haar cascade and zooms in if they are far away."""
    if frame is None:
        return None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return frame # No faces detected, return original
        
    # Find bounding box for all detected faces
    x_min = min([x for x, y, w, h in faces])
    y_min = min([y for x, y, w, h in faces])
    x_max = max([x + w for x, y, w, h in faces])
    y_max = max([y + h for x, y, w, h in faces])
    
    # Calculate average face width to determine if they are "far away"
    avg_width = np.mean([w for x, y, w, h in faces])
    
    # If faces are small (far away), perform digital zoom
    if avg_width < 120:
        print("   (Auto-Zoom Applied on far away faces)")
        
        # Add padding (e.g. 100% of the width/height to give context)
        padding_x = int((x_max - x_min) * 1.0)
        padding_y = int((y_max - y_min) * 1.0)
        
        # Add a minimum padding just in case
        padding_x = max(padding_x, 50)
        padding_y = max(padding_y, 50)
        
        h_orig, w_orig = frame.shape[:2]
        
        crop_x1 = max(0, x_min - padding_x)
        crop_y1 = max(0, y_min - padding_y)
        crop_x2 = min(w_orig, x_max + padding_x)
        crop_y2 = min(h_orig, y_max + padding_y)
        
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize back to original dimensions for the "zoom" effect
        zoomed_frame = cv2.resize(cropped_frame, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        return zoomed_frame
        
    return frame

def enhance_for_recognition(frame: np.ndarray) -> np.ndarray:
    """Balanced enhancement (Zoom -> Brighten -> Sharpen)"""
    if frame is None:
        return None
    
    # 1. Auto-Zoom if faces are far away
    frame = auto_zoom(frame)
    
    # 2. Add Auto-Brighten
    frame = auto_brighten(frame)
    
    # Ensure it's a color image
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # 3. Gentle Unsharp Mask
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(frame, 2.5, blurred, -1.5, 0)
    
    # 4. Light Laplacian
    lap = cv2.Laplacian(sharpened, cv2.CV_64F, ksize=3)
    sharpened = cv2.addWeighted(sharpened, 1.0, np.clip(lap, 0, 255).astype(np.uint8), 0.8, 0)
    
    # Final noise reduction
    final = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
    
    return final