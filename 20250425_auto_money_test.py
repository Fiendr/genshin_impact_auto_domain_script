from lg_mouse_controller import *
import time
import os
import cv2
from fiend_auto import *
import pyautogui as pa
from pynput.keyboard import Controller, Key

kb = Controller()

# 一圈大概3710像素

M = MoveR()



# leyline_completed_img_mat = cv2.imread(os.path.join("genshin_script_asset", "leyline_completed2.png"))

# while True:
#     result = find_img(leyline_completed_img_mat, (0, 0, 1920, 1080), 0.05)
#     print(result)
#     time.sleep(1)

def mouse_move_simulate(x, y):
    if os.name == "posix":
        # 模拟鼠标移动：
        device.emit(uinput.REL_X, x)
        device.emit(uinput.REL_Y, y)
        print("mouse simulate x, y", x, y)
    elif os.name == "nt":
        M.move(int(x), int(y))



mouse_move_simulate(-1860, 0)
time.sleep(1)
kb.press("w")
time.sleep(18)
kb.release("w")
time.sleep(1)

# mouse_move_simulate(152, 0)
# time.sleep(1)
# kb.press("w")
# time.sleep(8.8)
# kb.release("w")
# time.sleep(1)

