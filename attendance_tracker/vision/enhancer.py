import cv2
import numpy as np

def super_sharpen(image: np.ndarray) -> np.ndarray:
    """Aggressive sharpening focused on maximum definition (no brightening)"""
    # Strong Unsharp Mask
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(image, 3.0, blurred, -2.0, 0)
    
    # Extra edge enhancement using Laplacian
    lap = cv2.Laplacian(sharpened, cv2.CV_64F, ksize=3)
    sharpened = cv2.addWeighted(sharpened.astype(np.uint8), 1.0, np.clip(lap, 0, 255).astype(np.uint8), 1.0, 0)
    
    # High-pass filter for crisp details
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    high_pass = cv2.filter2D(sharpened, -1, kernel)
    
    # Final edge-preserving smooth to reduce noise
    final = cv2.bilateralFilter(high_pass, d=5, sigmaColor=50, sigmaSpace=50)
    
    return np.clip(final, 0, 255).astype(np.uint8)

def enhance_for_recognition(frame: np.ndarray) -> np.ndarray:
    """Main function - Heavy sharpening only (no brightening)"""
    if frame is None:
        return None
    
    # Work on a copy
    enhanced = frame.copy()
    
    # Apply super sharpening
    enhanced = super_sharpen(enhanced)
    
    return enhanced