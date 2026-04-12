import cv2
import numpy as np

def enhance_for_recognition(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return None
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Gentle sharpening
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(frame, 2.5, blurred, -1.5, 0)
    
    final = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
    return final