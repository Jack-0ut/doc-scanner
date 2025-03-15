from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders four points in top-left, top-right, bottom-right, bottom-left order.
    
    Args:
        pts (np.ndarray): Array of four points.

    Returns:
        np.ndarray: Ordered points.
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
    right_most = x_sorted[2:, :]
    
    tl, bl = left_most
    d = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    br, tr = right_most[np.argsort(d)[::-1], :]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applies a perspective transformation to obtain a top-down view of an image.
    
    Args:
        image (np.ndarray): Input image.
        pts (np.ndarray): Array of four points defining the region to transform.

    Returns:
        np.ndarray: Warped image with a top-down view.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect
    
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))
