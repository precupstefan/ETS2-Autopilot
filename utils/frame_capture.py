import numpy as np
import win32gui
from PIL import ImageGrab

windows_list = []
toplist = []
game_hwnd = 0


def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))


def get_window_handle(window_name):
    global game_hwnd
    win32gui.EnumWindows(enum_win, toplist)
    for (hwnd, win_text) in windows_list:
        if "Euro Truck Simulator 2" in win_text:
            game_hwnd = hwnd


def get_window_dead_zone(handle):
    # TODO: implement actuall stuff using win32gui, dont hardcode it
    return (9, 33, 9,9)


def get_window_frame_with_removed_borders(handle):
    position = win32gui.GetWindowRect(handle)
    dead_zone = get_window_dead_zone(handle)
    actual_pos = (
        position[0] + dead_zone[0],
        position[1] + dead_zone[1],
        position[2] - dead_zone[2],
        position[3] - dead_zone[3])
    return actual_pos
    window_attributes = win32gui.GetWindowDC(handle)


def get_screen_grab_of_window(window_name="Euro Truck Simulator 2"):
    if game_hwnd == 0:
        get_window_handle(window_name)
    position = get_window_frame_with_removed_borders(game_hwnd)
    screenshot = ImageGrab.grab(position, all_screens=True)
    screenshot = np.array(screenshot)
    return screenshot
