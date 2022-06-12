import numpy as np

from config import config


class Line:

    is_present = False
    is_continuous = False
    line_fit = None
    line_fit_m = None
    curvature_radius = 0
    orientation = 0
    points = None
    line_color = None

    def __init__(self, height, line_color):
        self.height = height
        self.line_color = line_color

    # line_fit represents parametric ecuation for pixels
    # line_fit_m represents parametric equation for meters
    def set_parameters(self, line_fit, line_fit_m):
        self.line_fit = line_fit
        self.line_fit_m = line_fit_m
        self.calculate_curvature_radius()
        self.compute_points()

    def calculate_curvature_radius(self):
        """
        left_fit and right_fit are assumed to have already been converted to meters
        """

        # meters per pixel in y dimension
        ym_per_pix = config["video"]["y_meters_per_pixel"]
        frame_height = config["video"]["size"][1]

        # y_eval is where we want to evaluate the fits for the line radius calcuation
        # for us it's at the bottom of the image for us, and because we know
        # the size of our video/images we can just hardcode it
        y_eval = frame_height * ym_per_pix
        fit = self.line_fit_m

        # https://stackoverflow.com/a/40021903
        if fit.size != 0:
            curve_rad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        else:
            curve_rad = None
        self.curvature_radius = curve_rad

    def get_fit_x(self, y):
        """
        y is a np.array
        @returns np.array
        """
        if self.line_fit_m.size == 0:
            return np.empty(y.shape)
        fit = self.line_fit
        return np.array(fit[0] * y ** 2 + fit[1] * y + fit[2]).astype("int")

    def get_angle(self, x):
        # see https://www.youtube.com/watch?v=uVLWZCPwTh8 for more info
        m = self.get_slope_m(x)
        return np.arctan(m) if m else None

    def get_slope_m(self, x):
        if self.line_fit_m.size == 0:
            return None
        return 2 * self.line_fit_m[0] * x + self.line_fit_m[1]

    def compute_points(self):
        ploty = np.linspace(0, self.height - 1, self.height).astype("int")
        if self.line_fit.size != 0:
            line_fit_x = self.get_fit_x(ploty)
            line = np.vstack((ploty, line_fit_x))
            # line = line[:, line[-1, :] < frame.shape[1]]
            self.points = np.transpose(line)[:, ::-1]
        else:
            self.points = np.empty((0,2))
