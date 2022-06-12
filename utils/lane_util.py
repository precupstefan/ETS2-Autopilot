import cv2

import numpy as np

from utils.image_util import draw_points


def draw_lane_info_on_perspective_image(lane, perspective_frame):
    frame = perspective_frame[:, ].copy()
    _draw_line_on_image(lane.left_line, frame, lane.left_line.line_color)
    _draw_line_on_image(lane.right_line, frame, lane.right_line.line_color)
    center_line = get_center_line_points(perspective_frame)
    draw_points(frame, center_line, [255,255,255]) #ALB
    middle_points = lane.get_center_line_points()
    draw_points(frame,middle_points,[0,255,0]) #verde
    return frame


def _draw_line_on_image(line, frame, color):
    if line.line_fit.size != 0:
        draw_points(frame, line.points, color)


def get_center_line_points(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    ploty = np.linspace(0, height - 1, height).astype("int")
    plotx = np.full(ploty.shape, int(width / 2))
    points = np.vstack((ploty, plotx))
    return np.transpose(points)[:, ::-1]
