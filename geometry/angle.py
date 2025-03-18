import numpy as np
import math

class AngleCalculator:
    @staticmethod
    def angle_between_vectors_degrees(u, v):
        """Returns the angle between two vectors in degrees"""
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    @staticmethod
    def get_angle(p1, p2, p3):
        """Returns the angle between the line segment from p2 to p1 and p2 to p3 in degrees"""
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b
        return AngleCalculator.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        """Returns the range between max and min interior angles of quadrilateral."""
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)
