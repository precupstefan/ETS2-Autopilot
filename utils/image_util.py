import cv2
import numpy as np

from config import config


def apply_threshold(frame, block_size, constant):
    """
    Replace each pixel in an image with a black pixel if the image intensity is less than some fixed constant T,
    or a white pixel if the image intensity is greater than that constant
    """
    # ret, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   block_size, constant)
    return thresh


def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_lines_using_hough_transform(frame):
    # TODO: Tune this parameters for better line detection
    return cv2.HoughLinesP(frame, rho=2, theta=np.pi / 180, threshold=40, minLineLength=25, maxLineGap=70)


def apply_gaussian_blur(frame, kernel_size=(5, 5), deviation=0):
    return cv2.GaussianBlur(frame, kernel_size, deviation)


def apply_canny(frame, low_threshold=50, high_threshold=150):
    return cv2.Canny(frame, low_threshold, high_threshold)


def fill_poly(frame, polygon, intensity=(255, 255, 255)):
    return cv2.fillPoly(frame, polygon, intensity)


def apply_bitwise_and(img1, img2):
    return cv2.bitwise_and(img1, img2)


def add_points(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0]  # Red
    thickness = -1
    radius = 15
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.circle(img2, (x0, y0), radius, color, thickness)
    cv2.circle(img2, (x1, y1), radius, color, thickness)
    cv2.circle(img2, (x2, y2), radius, color, thickness)
    cv2.circle(img2, (x3, y3), radius, color, thickness)
    return img2


def add_lines(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0]  # Red
    thickness = 2
    if src.shape[1] == 4:
        for line in src:
            x0, y0, x1, y1 = line
            cv2.line(img2, (x0, y0), (x1, y1), color, thickness)
    else:
        x0, y0 = src[0]
        x1, y1 = src[1]
        x2, y2 = src[2]
        x3, y3 = src[3]
        cv2.line(img2, (x0, y0), (x1, y1), color, thickness)
        cv2.line(img2, (x1, y1), (x2, y2), color, thickness)
        cv2.line(img2, (x2, y2), (x3, y3), color, thickness)
        cv2.line(img2, (x3, y3), (x0, y0), color, thickness)
    return img2


def draw_ROI_over_image(frame):
    src = np.array(config["LKAS"]["lanes_detection"]["warp_perspective"]["src"])
    aux = src[2].copy()
    src[2] = src[3]
    src[3] = aux
    return add_lines(frame, src)


def warp_perspective(frame):
    img_size = (frame.shape[1], frame.shape[0])
    src = np.float32([
        config["LKAS"]["lanes_detection"]["warp_perspective"]["src"]
    ])
    dst = np.float32([
        config["LKAS"]["lanes_detection"]["warp_perspective"]["dst"]
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(frame, matrix, img_size), m_inv


def draw_points(frame, points, color):
    for point in points:
        cv2.circle(frame, tuple(point), radius=2, color=color, thickness=-1)
