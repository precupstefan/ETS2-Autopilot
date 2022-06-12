import numpy as np

from LKAS.config import load_config, config
from LKAS.models.direction import Direction


class Trajectory:
    def __init__(self):
        pass

    def compute_trajectory(self, lane):
        """
        Returns the curvature in degrees
        """
        left_line = lane.left_line
        right_line = lane.right_line
        left_curvature = left_line.curvature_radius
        right_curvature = right_line.curvature_radius

        if left_curvature is None and right_curvature is None:
            return None

        direction = None
        point_of_interest = 0
        if left_curvature is None and right_curvature is not None:
            avg_curvature_radius = right_curvature
            m_avg = right_line.get_slope_m(point_of_interest)
            direction = Direction.LEFT if m_avg > 0 else Direction.RIGHT
            steering_value = m_avg * -1

        if left_curvature is not None and right_curvature is None:
            avg_curvature_radius = left_curvature
            m_avg = left_line.get_slope_m(point_of_interest)
            direction = Direction.RIGHT if m_avg > 0 else Direction.LEFT
            steering_value = m_avg

        if left_curvature is not None and right_curvature is not None:
            avg_curvature_radius = np.average([left_curvature, right_curvature])
            left_orientation = left_line.get_slope_m(point_of_interest)
            right_orientation = right_line.get_slope_m(point_of_interest)
            m_avg = (left_orientation + right_orientation) / 2
            slope_threshold = config["LKAS"]["lanes_detection"]["others"]["slope_difference"]
            if m_avg > slope_threshold:
                direction = Direction.LEFT
                steering_value = m_avg * -1
            elif m_avg < -slope_threshold:
                direction = Direction.RIGHT
                steering_value = m_avg * -1
            else:
                direction = Direction.STRAIGHT
                steering_value = 0

        #TODO: figure out how to use offcenter
        off_center = lane.get_off_center_of_lane()
        off_center = 0
        print("curvature", avg_curvature_radius)
        print("deviation", off_center)
        print("slope", m_avg)

        steering_value = steering_value - off_center
        print("steering_value", steering_value)

        return np.rad2deg(np.arctan(steering_value))
