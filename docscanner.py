from image_processing_utils import transform
from image_processing_utils import utils 
from matplotlib.patches import Polygon
import polygon_editor as poly_i
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
from geometry.corner import CornerDetector
from geometry.angle import AngleCalculator
import os

class DocScanner(object):
    """An image scanner"""

    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        """
        Args:
            interactive (boolean): If True, user can adjust screen contour before
                transformation occurs in interactive pyplot window.
            MIN_QUAD_AREA_RATIO (float): A contour will be rejected if its corners 
                do not form a quadrilateral that covers at least MIN_QUAD_AREA_RATIO 
                of the original image. Defaults to 0.25.
            MAX_QUAD_ANGLE_RANGE (int):  A contour will also be rejected if the range 
                of its interior angles exceeds MAX_QUAD_ANGLE_RANGE. Defaults to 40.
        """        
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE
        self.corner_detector = CornerDetector()        

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """Returns True if the contour satisfies all requirements set at instantitation"""

        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and AngleCalculator().angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

    def get_contour(self, rescaled_image):
        """Returns the contour representing the document in the image"""
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.corner_detector.get_corners(edged)

        approx_contours = []
        if len(test_corners) >= 4:
            quads = []
            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype="int32")
                quads.append(points)

            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            quads = sorted(quads, key=lambda quad: AngleCalculator().angle_range(quad))

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)

        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        """Allows the user to manually adjust the document corners in an interactive UI"""
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title('Drag the corners of the box to the corners of the document. \nClose the window when finished.')
        p = poly_i.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype="int32")
        return new_points.reshape(4, 2)

    def scan(self, image_path):
        """Main scanning function"""
        RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = '.'

        image = cv2.imread(image_path)
        assert(image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = utils.resize(image, height=int(RESCALED_HEIGHT))

        screenCnt = self.get_contour(rescaled_image)
        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        warped = transform.four_point_transform(orig, screenCnt * ratio)

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        basename = os.path.basename(image_path)
        cv2.imwrite(OUTPUT_DIR + '/' + basename, thresh)
        return thresh  # Return the scanned image
