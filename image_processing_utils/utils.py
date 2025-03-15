import numpy as np
import cv2

def translate(image: np.ndarray, x: int, y: int) -> np.ndarray:
    """Shifts an image by (x, y) pixels.
    
    Args:
        image (np.ndarray): Input image.
        x (int): Shift along the x-axis.
        y (int): Shift along the y-axis.

    Returns:
        np.ndarray: Translated image.
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def rotate(image: np.ndarray, angle: float, center: tuple = None, scale: float = 1.0) -> np.ndarray:
    """Rotates an image around its center.
    
    Args:
        image (np.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        center (tuple, optional): Center of rotation. Defaults to image center.
        scale (float, optional): Scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Rotated image.
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))


def resize(image: np.ndarray, width: int = None, height: int = None, inter: int = cv2.INTER_AREA) -> np.ndarray:
    """Resizes an image while maintaining aspect ratio.
    
    Args:
        image (np.ndarray): Input image.
        width (int, optional): Desired width. Defaults to None.
        height (int, optional): Desired height. Defaults to None.
        inter (int, optional): Interpolation method. Defaults to cv2.INTER_AREA.

    Returns:
        np.ndarray: Resized image.
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)
