import cv2
from angle import AngleCalculator

class ContourValidator:
    def __init__(self, min_area_ratio=0.25, max_angle_range=40):
        self.min_area_ratio = min_area_ratio
        self.max_angle_range = max_angle_range

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """Returns True if the contour satisfies all requirements"""
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.min_area_ratio
                and self.angle_range(cnt) < self.max_angle_range)

    def angle_range(self, quad):
        """Returns the range of angles for the quadrilateral"""
        # Uses the AngleCalculator to compute the angle range
        angle_calculator = AngleCalculator()
        return angle_calculator.angle_range(quad)
