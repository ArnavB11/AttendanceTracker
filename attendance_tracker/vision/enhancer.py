import cv2
import numpy as np

def enhance_for_recognition(frame: np.ndarray) -> np.ndarray:
    """Balanced sharpening - Clear but not destructive"""
    if frame is None:
        return None
    
    # Ensure it's a color image
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Gentle Unsharp Mask
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(frame, 2.5, blurred, -1.5, 0)
    
    # Light Laplacian
    lap = cv2.Laplacian(sharpened, cv2.CV_64F, ksize=3)
    sharpened = cv2.addWeighted(sharpened, 1.0, np.clip(lap, 0, 255).astype(np.uint8), 0.8, 0)
    
    # Final noise reduction
    final = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
    
    return final