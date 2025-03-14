import numpy as np
import cv2

def order_points(pts):
    """
    Orders a set of four points in the following order:
    - Top-left
    - Top-right
    - Bottom-right
    - Bottom-left

    This ensures a consistent ordering for perspective transformations.

    Parameters:
        pts (numpy.ndarray): Array of shape (4, 2) containing the four (x, y) coordinates.

    Returns:
        numpy.ndarray: Ordered array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Compute sums and differences of points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Assign points based on sum and difference calculations
    rect[0] = pts[np.argmin(s)]  # Top-left has the smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has the largest sum
    rect[1] = pts[np.argmin(diff)]  # Top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has the largest difference
    
    return rect

def four_point_transform(image, pts):
    """
    Applies a four-point perspective transform to warp an image into a top-down view.

    Parameters:
        image (numpy.ndarray): Input image.
        pts (numpy.ndarray): Array of four points (x, y) defining the region to be transformed.

    Returns:
        numpy.ndarray: Warped image with a bird's-eye view.
    """
    # Ensure the points are in the correct order
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the transformed image
    widthA = np.linalg.norm(br - bl)  # Bottom width
    widthB = np.linalg.norm(tr - tl)  # Top width
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the transformed image
    heightA = np.linalg.norm(tr - br)  # Right height
    heightB = np.linalg.norm(tl - bl)  # Left height
    maxHeight = max(int(heightA), int(heightB))

    # Define destination points for the top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply the transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped
