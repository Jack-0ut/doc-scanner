from transform import four_point_transform
from skimage.filters import threshold_local
import cv2
import imutils
import numpy as np

def process_image(image):
    """Processes the input image for document scanning and returns the transformed images."""
    # Ensure the image is not None
    if image is None:
        raise ValueError("Error: Invalid image input.")

    # Copy the original image to preserve it
    orig = image.copy()

    # Compute scale ratio for resizing for edge detection purposes
    ratio = image.shape[0] / 500.0
    image_resized = imutils.resize(image, height=500)  # Resize for edge detection

    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # Find contours on the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours by area to keep the largest ones
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 0.02 is the precision of approximation
        
        # If the contour has four points, assume it's the document
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        raise ValueError("Error: Could not find a document outline.")

    # Adjust contour points for the resized image
    screenCnt = screenCnt.reshape(4, 2) * ratio  # Scale back to original image size
    warped = four_point_transform(orig, screenCnt)  # Apply the perspective transform

    # Convert the warped image to grayscale and apply thresholding
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped_gray, 11, offset=10, method="gaussian")
    warped_thresh = (warped_gray > T).astype("uint8") * 255

    return image_resized, thresh, orig, warped_thresh
