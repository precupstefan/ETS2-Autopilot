# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

from modules.lane_detection.lane_detection import LaneDetection
from utils.frame_capture import get_screen_grab_of_window

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lkas = LaneDetection()
    while True:
        screenshot = get_screen_grab_of_window()
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        lkas.detect_lanes(screenshot)
        # cv2.imshow("Screen", screenshot)
        cv2.waitKey(1)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
