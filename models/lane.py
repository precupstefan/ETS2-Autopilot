import numpy as np

import cv2

from config import config
from models.line import Line


class Lane:

    def __init__(self, height):
        self.left_line = Line(height, [255, 0, 0])
        self.right_line = Line(height, [0, 0, 255])
        self.middle_lane_x = None

    def set_middle_lane_x(self, middle_lane_x):
        self.middle_lane_x = middle_lane_x

    def get_off_center_of_lane(self):
        middle_point = self.get_middle_point_for_y(np.array([720]))
        xm_per_pix = config["video"]["x_meters_per_pixel"]
        frame_width = config["video"]["size"][0]
        pixel_deviation = frame_width / 2 - abs(middle_point)
        deviation = pixel_deviation * xm_per_pix
        return deviation[0]

    def get_middle_point_for_y(self, y):
        """
        @param y: numpy array
        @return: np.array
        """
        left_x = self.left_line.get_fit_x(y)
        right_x = self.right_line.get_fit_x(y)
        return np.mean([left_x, right_x], axis=0)

    def get_center_line_points(self):
        if self.left_line.points.size == 0 and self.right_line.points.size == 0:
            return np.empty((0, 2))
        if self.left_line.points.size == 0:
            return self.right_line.points
        elif self.right_line.points.size == 0:
            return self.left_line.points
        return np.average([self.left_line.points, self.right_line.points], axis=0).astype("int")
