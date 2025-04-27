# 前置准备:
#   队伍只有一个角色
#   没有新手任务导航提示
#   窗口分辨率1280x720默认位置
#
# 测试硬件:
#   cpu AMD 3500X               # 推理耗时 1000ms   3600 500ms
#   gpu AMD RTX 6700XT rocm6.3  # 推理耗时 40ms
#   ubuntu 22.04.5  X11 and win11

import threading
import mss
import torch
from ultralytics import YOLO
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QEventLoop
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, \
    QPushButton
import os
import cv2
import math

import pynput
from pynput.mouse import Button
from pynput.keyboard import Key
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
from queue import Queue


""" 跨平台代码 """
if os.name == "posix":
    import uinput
    # 驱动级键鼠模拟
    events = (
        uinput.REL_X,
        uinput.REL_Y,
        uinput.BTN_LEFT,
        uinput.BTN_MIDDLE,
        uinput.BTN_RIGHT,
    )
    # 全局创建 uinput 设备
    device = uinput.Device(events, name="virtual-mouse2")
elif os.name == "nt":
    from lg_mouse_controller import *
    M = MoveR()

""" 全局变量 """
ASSET_DIR = "genshin_script_asset"
AUTO_FIGHT_SCRIPT_DIR = "auto_fight_script"
AUTO_FIGHT_SCRIPT_KEY_LIST = []     # 出招表
FLAG = None # 地脉之 "money" or "experience"
IS_FIND_BLOOD_BAR = False

show_tree_img_queue = Queue(1000)
show_minimap_img_queue = Queue(1000)
show_statu_queue = Queue(20)
send_to_main_queue = Queue(5)
send_to_walk_queue = Queue(5)

DOMAIN_NAME = ""

import pyautogui as pa
from fiend_auto import *
import pygame

pygame.init()
pygame.mixer.init()

thread_pool_executor = ThreadPoolExecutor(max_workers=10)

kb = pynput.keyboard.Controller()
mouse = pynput.mouse.Controller()


yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
# YOLO init 原神'绝缘草本水本火本'2800张手动标注 100epochs
model_path = os.path.join(ASSET_DIR, "原神绝缘草本水本火本2800张100epochs_best.pt")
model = YOLO(model_path)  # 禁用详细日志m
model = model.to(yolo_device)

# 基于1280 x 720
region_x, region_y, region_width, region_height = 320, 192, 1280, 340
minimap_x, minimap_y, minimap_width, minimap_height = 361, 204, 142, 142
img_center_x = region_width // 2  # 相对于图片中心X坐标

IS_CHANGED_SHUZI = False  # 是否换过树脂, 每次接收translate时判断
DOMAIN = "绝缘本"
                                                                                                              
# DOMAIN = "猎人本"
ACCOUNT_INDEX = 1


def play_mp3(path):
    # 使用绝对路径
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    # 循环等待直到音乐播放结束
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# 音频
mp3_nomore_shuzi = os.path.join(ASSET_DIR, "nomorshuzi.mp3")

"坐标"
pos_map_all = 1548, 869
pos_map_mengDe = 1287, 305
pos_map_daoQi = 1292, 376
pos_map_fendang = 1283, 445
pos_map_fendang_domain1 = 1106, 849
pos_map_fendang_domain2 = 753, 709
pos_map_fendang_domain_cangbaideyirong = 1128,885   # 苍白的遗容
pos_map_fendan_top_qitianshenxiang = 1126,324   # 枫丹最上面七天神像
pos_map_nata = 1470, 421
pos_map_nata_domain = 940, 692
pos_map_nata_domain_xialuoben = 564, 798


pos_map_reduce = 349, 612  # map缩小 306,618
pos_jueYuanBen = 427, 737

pos_translate = 1302, 861

pos_map_zoom = 349, 484  # map放大     307,482
pos_map_daoqi_translate_point = 957, 547  # 稻妻主城传送点

pos_pipei = 1160, 866  # 匹配
pos_confirm = 1382, 868  # 多人挑战
pos_confirm2 = 991, 694  # 多人挑战 -> 确认
pos_team_cancel = 1547, 221  # 组队界面叉叉

pos_quit_challenge = 845, 867  # 死亡状态 放弃挑战
# 结算界面
pos_shuzi = 722, 694  # 树脂
pos_challenge_continue = 1003, 855  # 继续挑战
pos_cancel_challenge = 717, 853  # 退出挑战

# 晶蝶
pos_qitianshengxiang_fengqidi = 1060, 637  # 七天神像-风起地
pos_lifeideliuxing_domain = 1282, 456  # 逆飞的流星本
pos_map_liyue = 1489, 310  # map 璃月

"图片"
# 匹配中...
img_pipeizhong = os.path.join(ASSET_DIR, "pipeizhong.bmp")
area_pipeizhong = 1146, 856, 1162, 882
img_pipeizhong_cancel = os.path.join(ASSET_DIR, "pipeizhong_cancel.bmp")
area_pipeizhong_cancel = 807, 673, 930, 720
img_2p_nobody = os.path.join(ASSET_DIR, "2P_nobody.bmp")
area_2p_nobody = 822, 518, 876, 577
# 派蒙
img_main_interface = os.path.join(ASSET_DIR, "main_interface.bmp")
area_main_interface = 346, 210, 372, 236
# 组队界面 + 号
img_team_plus = os.path.join(ASSET_DIR, "team_plus.bmp")
area_team_plus = 1099, 841, 1143, 896
# 放弃挑战 死亡
img_cancel_challenge = os.path.join(ASSET_DIR, "cancel_challenge.bmp")
area_cancel_challenge = 706, 843, 732, 872
# N秒后自动退出 通关
img_auto_out = os.path.join(ASSET_DIR, "completed.bmp")
area_auto_out = 923, 817, 1038, 843
# 退出秘境
img_out_domain = os.path.join(ASSET_DIR, "out_domain.bmp")
area_out_domain = 704, 832, 899, 881
# 树脂耗尽
img_nomore_shuzi = os.path.join(ASSET_DIR, "img_nomore_shuzi.bmp")
area_nomore_shuzi = 709, 682, 736, 713
# 副本首页tips  "点击任意位置关闭"
img_domain_tips = os.path.join(ASSET_DIR, "domain_tips.png")
area_domain_tips = 878, 641, 1024, 667
# F
img_f = os.path.join(ASSET_DIR, "f.png")
img_f_mat = cv2.imread(img_f)
area_f = 1093 - 40, 641 - 101, 1117 - 40, 661 - 101
# 地脉
img_money = os.path.join(ASSET_DIR, "money.png")
img_money_mat = cv2.imread(img_money)
img_experience = os.path.join(ASSET_DIR, "experience.png")
img_experience_mat = cv2.imread(img_experience)
region_minimap = minimap_x, minimap_y, minimap_x+minimap_width, minimap_y+minimap_height
minimap_center_x, minimap_center_y = minimap_width//2, minimap_height//2
# 地脉完成挑战宝箱状图片
leyline_completed_img_mat = cv2.imread(os.path.join("genshin_script_asset", "leyline_completed2.png"))
img_leylinereward_mat = cv2.imread(os.path.join("genshin_script_asset", "leylinereward.png"))   # 领取地脉奖励叉号
region_leylinereward = 1201,372,1232,404  # 领取地脉奖励叉号 匹配区域
img_blood_bar_mat = cv2.imread(os.path.join(ASSET_DIR, "blood_bar.png"))
region_game = 321,193,1595,906  


def finding_main_interface():
    for i in range(60):
        time.sleep(1)
        result_mian_interface = find_pic(img_main_interface, area_main_interface, 0.095)
        print("找派蒙:", result_mian_interface)
        if result_mian_interface[0]:
            time.sleep(1)
            return True
    return False


def catch_jingdie():
    time.sleep(2)
    # 风起地
    for x in range(2):
        finding_main_interface()
        time.sleep(1)
        pa.press("m")
        time.sleep(1)
        pa.click(pos_map_all)
        time.sleep(1)
        pa.click(pos_map_mengDe)
        time.sleep(1)
        for i in range(7):
            pa.click(pos_map_reduce)
            time.sleep(0.2)
        time.sleep(0.4)
        pa.click(pos_qitianshengxiang_fengqidi)
        time.sleep(1)
        pa.click(pos_translate)
        finding_main_interface()
        time.sleep(1)
        pa.press("s")
        time.sleep(1)
        pa.click(button="middle")
        time.sleep(1)
        if x == 0:
            # 右转1秒
            pa.keyDown("d")
            time.sleep(1)
            pa.keyUp("d")
        if x == 1:
            # 左转0.5秒
            pa.keyDown("a")
            time.sleep(0.7)
            pa.keyUp("a")
        # 前进
        time.sleep(0.2)
        pa.keyDown("w")
        #   space + f
        for i in range(50):
            time.sleep(0.1)
            pa.press("space")
            pa.press("f")
        pa.keyUp("w")
        time.sleep(1)

    # 逆飞的流星本
    finding_main_interface()
    time.sleep(1)
    pa.press("m")
    time.sleep(1)
    pa.click(pos_map_all)
    time.sleep(1)
    pa.click(pos_map_liyue)
    time.sleep(1)
    for i in range(7):
        time.sleep(0.2)
        pa.click(pos_map_reduce)
    time.sleep(1)
    pa.click(pos_lifeideliuxing_domain)
    time.sleep(1)
    pa.click(pos_translate)
    finding_main_interface()
    time.sleep(1)
    pa.press("s")
    time.sleep(0.2)
    pa.click(button="middle")
    time.sleep(1)
    pa.keyDown("a")
    time.sleep(0.2)
    pa.keyUp("a")
    time.sleep(0.2)
    pa.keyDown("w")
    for i in range(10):
        time.sleep(0.1)
        pa.press("space")
        pa.press("f")
    time.sleep(0.2)
    pa.keyUp("w")
    time.sleep(1)
    # 右转
    pa.keyDown("d")
    time.sleep(0.1)
    pa.keyUp("d")
    time.sleep(1)
    pa.click(button="middle")
    time.sleep(1)
    pa.keyDown("w")
    time.sleep(0.2)
    for i in range(18):
        time.sleep(0.1)
        pa.press("space")
        pa.press("f")
    time.sleep(0.2)
    pa.keyUp("w")
    time.sleep(1)


def change_shuzi():
    time.sleep(2)
    pa.press("m")
    time.sleep(1)
    pa.click(pos_map_all)
    time.sleep(1)
    pa.click(pos_map_mengDe)
    time.sleep(1)
    pa.click(pos_map_all)
    time.sleep(1)
    pa.click(pos_map_daoQi)
    time.sleep(1)
    for i in range(6):
        pa.click(pos_map_zoom)
        time.sleep(0.4)
    time.sleep(1)
    pa.click(pos_map_daoqi_translate_point)
    time.sleep(1)
    pa.click(pos_translate)
    finding_main_interface()
    time.sleep(2)
    # 前进10秒
    pa.keyDown("w")
    time.sleep(11.6)
    pa.keyUp("w")
    time.sleep(0.5)
    # 左转2秒
    pa.keyDown("a")
    time.sleep(1.2)
    pa.keyUp("a")
    time.sleep(1)
    for i in range(2):
        pa.press("f")
        time.sleep(1)
    # 瑶瑶短腿, 再加1秒
    pa.keyDown("w")
    time.sleep(1)
    pa.keyUp("w")
    time.sleep(0.5)
    for i in range(2):
        pa.press("f")
        time.sleep(1)

    time.sleep(2)
    pa.click(1412, 867)  # 合成
    time.sleep(2)
    pa.click(868, 791)  # 确定
    time.sleep(2)
    pa.click(881, 692)  # 没有树脂的情况, 取消
    time.sleep(2)
    pa.click(1548, 221)  # 叉叉
    time.sleep(2)


""" 绝缘本 """


def translate_to_jueyuanben():
    time.sleep(2)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_daoQi)
    time.sleep(2)
    for i in range(6):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    pa.click(pos_jueYuanBen)
    time.sleep(2)
    pa.click(pos_translate)
    # 找派蒙
    finding_main_interface()


""" 火本 """


def translate_to_huoben():
    time.sleep(2)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_nata)
    time.sleep(2)
    for i in range(6):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    pa.click(pos_map_nata_domain)
    time.sleep(2)
    pa.click(pos_translate)
    # 找派蒙
    finding_main_interface()


""" 猎人本 """


def translate_to_lierenben():
    time.sleep(3)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_fendang)
    time.sleep(2)
    for i in range(8):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    # 传送副本1
    pa.click(pos_map_fendang_domain1)
    time.sleep(2)
    pa.click(pos_translate)
    finding_main_interface()  # 等待加载
    time.sleep(2)
    # 传送副本2 即猎人本
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_fendang_domain2)
    time.sleep(2)
    pa.click(pos_translate)
    finding_main_interface()  # 等待加载

def translate_to_xialuoben():
    time.sleep(3)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    print("click pos map nata")
    pa.click(pos_map_nata)
    time.sleep(2)
    for i in range(8):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    # 传送副本1 火本
    pa.click(pos_map_nata_domain)
    time.sleep(2)
    pa.click(pos_translate)
    finding_main_interface()  # 等待加载
    time.sleep(2)
    # 传送副本2 即下落本
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_nata_domain_xialuoben)
    time.sleep(2)
    pa.click(pos_translate)
    finding_main_interface()  # 等待加载


"""
按M开地图 传送 一直到进本
"""


def translate():
    time.sleep(1)
    pa.leftClick(1920 // 2, 1080 // 2)
    for j in range(20):
        # 是否换树脂
        global IS_CHANGED_SHUZI
        if not IS_CHANGED_SHUZI:
            # print("抓晶蝶...")
            # catch_jingdie()
            time.sleep(1)
            print("换树脂...")
            change_shuzi()
            IS_CHANGED_SHUZI = True

        if DOMAIN == "雷本":
            translate_to_jueyuanben()
        # elif DOMAIN == "草本":
        #     translate_to_lierenben()
        elif DOMAIN == "水本":
            translate_to_lierenben()
        elif DOMAIN == "火本":
            translate_to_huoben()
        elif DOMAIN == "下落本":
            translate_to_xialuoben()
        elif DOMAIN == "":
            print("DOMAIN empty errror")
            exit()

        # 前进 一次
        pa.keyDown("w")
        time.sleep(2)
        pa.keyUp("w")
        time.sleep(1)
        for i in range(3):
            pa.press("f")
            time.sleep(0.2)
        # 后退 三次
        for i in range(3):
            pa.keyDown("s")
            time.sleep(1)
            pa.keyUp("s")
            time.sleep(1)
            for k in range(3):
                pa.press("f")
                time.sleep(0.2)

        # 匹配q
        while True:
            pa.click(pos_pipei)
            time.sleep(2)
            result_pipeizhong_cancel = find_pic(img_pipeizhong_cancel, area_pipeizhong_cancel, 0.1)
            print("result_pipeizhong_cancel:", result_pipeizhong_cancel)
            time.sleep(1)
            if result_pipeizhong_cancel[0]:
                pa.click(result_pipeizhong_cancel[0], result_pipeizhong_cancel[1])
                time.sleep(1)
            else:
                print("break")
                break

        # 等待匹配结束
        while True:
            time.sleep(5)
            result_a = find_pic(img_pipeizhong, area_pipeizhong, 0.29)  # 0.31 未找到 0.26找到
            print("pipeizhong", result_a)
            if result_a[0]:
                continue
            else:
                break

        # 邀请
        time.sleep(14)
        pa.click(pos_confirm)
        time.sleep(1)
        pa.click(pos_confirm2)
        time.sleep(5)

        # 等待队友同意
        count = 0
        while count < 15:
            time.sleep(1)
            result_2P_nobody = find_pic(img_2p_nobody, area_2p_nobody, 0.03)
            print("result_2P_nobody: ", result_2P_nobody)
            if result_2P_nobody[0]:
                count += 1
                continue
            else:
                break  # 跳出 while

        if count == 15:
            for i in range(2):  # 点击两次叉号, 退出组队界面
                pa.click(pos_team_cancel)
                time.sleep(2)
                # 确认解散队伍
                pa.click(1101, 693)
                time.sleep(1)

            time.sleep(5)
            continue  # 队友没齐, 重新传送
        else:
            for i in range(20):
                time.sleep(2)
                for l in range(2):
                    pa.click(pos_confirm)
                    time.sleep(1)
                result_plus = find_pic(img_team_plus, area_team_plus, 0.1)
                if result_plus[0]:
                    pass
                else:
                    return 1  # 没有 + 号 则已进入副本
    return 0


"""
通关 return True
超3分钟 未通关 return False
"""


def fight():
    time.sleep(1)
    pa.press("e")
    time.sleep(1)
    for i in range(90):  # 4分钟
        time.sleep(1)
        # 每2秒找一次图
        result_completed = find_pic(img_auto_out, area_auto_out, 0.1)  # 0.1以下
        print("result_completed", result_completed)
        time.sleep(0.2)
        pa.press("e")
        time.sleep(0.2)
        pa.press("q")
        time.sleep(0.2)
        if result_completed[0]:
            return "True"  # 通关
    # 未通关 退出
    time.sleep(1)
    pa.click(850, 870)
    time.sleep(2)
    pa.click(1105, 692)
    time.sleep(20)
    return "False"


"""
返回True则已继续挑战
返回False则没有, 且已退出
"""


def challenge_continue():
    time.sleep(1)
    pa.click(pos_shuzi)
    time.sleep(18)
    for i in range(3):  #
        pa.click(pos_challenge_continue)
        time.sleep(1)
        # 查看是否树脂用完
        result_nomore_shuzi = find_pic(img_nomore_shuzi, area_nomore_shuzi, 0.1)
        if result_nomore_shuzi[0]:
            print("树脂用完了!!!\n" * 5)
            pa.click(result_nomore_shuzi[0], result_nomore_shuzi[1])  # 点击取消退出
            # 语音提示
            play_mp3(mp3_nomore_shuzi)
            time.sleep(3)
            print("done")
            return "done"

        time.sleep(13)
        result = find_pic(img_cancel_challenge, area_cancel_challenge, 0.1)  # 0.13 找到了
        print("img_cancel_challenge:", result)
        if result[0]:
            break
        else:
            return "True"
    time.sleep(2)
    pa.click(pos_cancel_challenge)
    # 找派蒙
    finding_main_interface()
    return "False"


def switch_account():
    global IS_CHANxGED_SHUZI
    # 切换账号后 重置
    IS_CHANGED_SHUZI = False

    finding_main_interface()
    time.sleep(1)
    pa.press("esc")
    time.sleep(1)
    pa.click(349, 873)
    time.sleep(1)
    # (960, 550)
    pa.click(960, 550)
    time.sleep(1)

    pa.click(990, 695)
    time.sleep(30)

    pa.click(1535, 846)  # 左箭头
    time.sleep(2)

    pa.click(1041, 642)  # 退出
    time.sleep(2)

    pa.click(1047, 570)
    time.sleep(15)
    pa.click(1117, 521)  # 账号下拉菜单
    time.sleep(1)
    if ACCOUNT_INDEX == 2:
        pa.click(807, 641)
    if ACCOUNT_INDEX == 3:
        pa.click(805, 700)
    time.sleep(1)
    for i in range(4):
        pa.click(963, 607)
        time.sleep(15)
    finding_main_interface()

""" 跨平台代码 """
def mouse_move_simulate(x, y):
    if os.name == "posix":
        # 模拟鼠标移动：
        device.emit(uinput.REL_X, x)
        device.emit(uinput.REL_Y, y)
        print("mouse simulate x, y", x, y)
    elif os.name == "nt":
        M.move(int(x), int(y))


def keyboard_press_simulate(key, delay):
    kb.press(key)
    time.sleep(delay)
    kb.release(key)


#################################################################################################

def screen_shot(x, y, width, height):
    with mss.mss() as sct:
        # monitor = {"top": 0, "left": 0, "width": 400, "height": 400}  用这个出错!得用下面括号里的
        screenshot = sct.grab({'left': x,
                               'top': y,
                               'width': width,
                               'height': height})
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
        return img


def minimap_rotation(target_angle=0):
    back_angle = (target_angle + 180) % 360

    while True:
        """ 小地图朝向角度检测 """
        img_minimap = screen_shot(minimap_x, minimap_y, minimap_width, minimap_height)
        current_angle = compute_mini_map_angle(img_minimap)
        print("current_angle", current_angle)
        img_minimap = draw_angle(img_minimap, current_angle)
        show_minimap_img_queue.put(img_minimap)
        abs_diff_angle = min(abs(target_angle-current_angle), 360-abs(target_angle-current_angle))  # 绝对差
        clockwise_diff_back_angle = (back_angle-current_angle+360) % 360    # 与背角 顺时针角度差
        move_distance = max(int(abs_diff_angle//8), 2)

        if abs_diff_angle <= 3:
            print("已朝向指定角度")
            break
        elif clockwise_diff_back_angle >= 180:  
            # 在左边, 向右移动
            mouse_move_simulate(move_distance, 0)
            print("angle right")
        elif clockwise_diff_back_angle < 180: 
            mouse_move_simulate(-move_distance, 0)
            print("angle left")

# 地脉领奖
def minimap_rotation2(current_angle, target_angle=270, tips="地脉领奖"):
    back_angle = (target_angle + 180) % 360

    abs_diff_angle = min(abs(target_angle-current_angle), 360-abs(target_angle-current_angle))  # 绝对差
    clockwise_diff_back_angle = (back_angle-current_angle+360) % 360    # 与背角 顺时针角度差
    move_distance = max(int(abs_diff_angle), 2)

    if abs_diff_angle <= 3:
        print("已朝向指定角度")
        return True
    elif clockwise_diff_back_angle < 180:  
        # 在左边, 向右移动
        mouse_move_simulate(move_distance, 0)
        print("angle right")
    elif clockwise_diff_back_angle >= 180: 
        mouse_move_simulate(-move_distance, 0)
        print("angle left")

    return False



def get_tree_difference():
    """ 古树检测 """
    img = screen_shot(region_x, region_y, region_width, region_height)
    time0 = time.time()
    results = model(img)
    time1 = time.time()

    boxes = results[0].boxes
    if boxes:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_center_x = int(x1 + (x2 - x1) / 2)  # 相对于图片
        difference = box_center_x - img_center_x
        print("difference", difference)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        img = cv2.putText(img, f"{(time1 - time0):.4f}", (5, 20),
                          cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                          1,
                          (255, 255, 0),
                          2)
        return difference, img

    return None, None


def walk_to_domain_center():
    results = model(np.zeros((1280, 340, 3), np.uint8))  # 预加载YOLO到内存
    print("start YOLO threading")
    while True:
        try:
            flag, _ = send_to_walk_queue.get(timeout=0.5)
            if flag == "start_yolo":
                print("start_yolo...")
                # 鼠标中键回正视角
                time.sleep(0.5)
                mouse.press(Button.middle)
                time.sleep(0.1)
                mouse.release(Button.middle)
                time.sleep(0.5)

                count = 0
                while True:
                    print("start minimap rotation thread")
                    # 视角朝东
                    thread1 = threading.Thread(target=minimap_rotation)
                    thread1.start()
                    thread1.join()
                    print("stop minimap rotation thread")
                    time.sleep(0.2)

                    # YOLO识别tree与center的距离
                    distance, img = get_tree_difference()
                    if not distance:
                        continue  # 未检测到tree

                    show_tree_img_queue.put(img)
                    print("difference:", distance)
                    delay = abs(distance) / 200
                    if distance > 15:
                        kb.release("a")
                        time.sleep(0.1)
                        kb.press("d")
                        time.sleep(delay)
                        kb.release("d")
                    elif distance < -15:
                        kb.release("d")
                        time.sleep(0.1)
                        kb.press("a")
                        time.sleep(delay)
                        kb.release("a")
                    else:
                        kb.release("d")
                        time.sleep(0.1)
                        kb.release("a")

                    count += 1
                    time.sleep(0.1)
                    print(count)
                    if count > 4:
                        kb.release("d")
                        kb.release("a")

                        for i in range(6 if yolo_device == "cuda" else 3):
                            # 视角朝向tree
                            distance, img = get_tree_difference()
                            show_tree_img_queue.put(img)
                            thread_pool_executor.submit(mouse_move_simulate, distance // 5, 0)
                        print("pause yolo...")
                        send_to_main_queue.put(("pause_yolo", None))
                        break
            elif flag == "end_yolo":
                print("end yolo...")
                break
        except:
            pass


def compute_mini_map_angle(mat):
    """
    计算当前小地图摄像机朝向的角度
    :param mat: 小地图灰度图
    :return: 角度（0-360度）
    """
    # 如果不是灰度图，转换成灰度图
    if len(mat.shape) == 3 and mat.shape[2] == 3:
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去噪
    mat = cv2.GaussianBlur(mat, (5, 5), 0)

    # 极坐标变换
    center = (mat.shape[1] / 2, mat.shape[0] / 2)  # 计算中心点
    polar_mat = cv2.warpPolar(mat, (360, 360), center, 360, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)

    # 提取极坐标 ROI 区域
    polar_roi_mat = polar_mat[:, 15:80]  # 取部分区域
    polar_roi_mat = cv2.rotate(polar_roi_mat, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 旋转90度

    # 计算梯度
    scharr_result = cv2.Scharr(polar_roi_mat, cv2.CV_32F, 1, 0)

    # 寻找波峰
    scharr_array = scharr_result.flatten()
    left_peaks, _ = find_peaks(scharr_array)
    right_peaks, _ = find_peaks(-scharr_array)  # 反向取波峰

    left = np.zeros(360)
    right = np.zeros(360)

    for i in left_peaks:
        left[i % 360] += 1
    for i in right_peaks:
        right[i % 360] += 1

    # 计算优化后的左右特征
    left2 = np.maximum(left - right, 0)
    right2 = np.maximum(right - left, 0)

    # 左移90度对齐并相乘
    sum_result = np.zeros(360)
    for i in range(-2, 3):
        shifted = np.roll(right2, -90 + i)
        sum_result += left2 * shifted * (3 - abs(i)) / 3

    # 卷积平滑
    result = np.zeros(360)
    for i in range(-2, 3):
        shifted = np.roll(sum_result, i) * (3 - abs(i)) / 3
        result += shifted

    # 找到最大值对应的角度
    angle = np.argmax(result) + 45
    if angle > 360:
        angle -= 360

    return angle


def draw_angle(img, angle, color=(0, 255, 255)):
    if angle:
        # 在原图上绘制直线
        center = (img.shape[1] // 2, img.shape[0] // 2)
        line_length = 100  # 直线长度
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        # 计算顺时针角度对应的直线终点（注意 y 坐标用加法）
        end_point = (
            int(center[0] + line_length * np.cos(angle_rad)),
            int(center[1] + line_length * np.sin(angle_rad))
        )
        # 使用红色（BGR: (0, 0, 255)）绘制直线，粗细为2
        img = cv2.line(img, center, end_point, (0, 0, 255), thickness=2)
        img = cv2.putText(img, str(angle), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img


def walk_to_f():
    count = 0
    kb.press("w")
    time.sleep(0.1)
    # mouse.press(Button.right)
    time.sleep(0.05)
    while True:
        count += 1
        if count > 1000:
            return "False"  # 10秒超时

        time.sleep(0.01)
        # pa.press("f")
        # time.sleep(0.01)
        result_f = find_img(img_f_mat, area_f, 0.04)
        print("result_f:", result_f)
        if result_f[0]:
            # mouse.release(Button.right)
            time.sleep(0.05)
            kb.release("w")
            time.sleep(0.05)
            kb.press("f")
            time.sleep(0.05)
            kb.release("f")
            return "True"


def find_domain_tips():
    count = 0
    while True:
        count += 1
        if count > 100:
            return "False"  # 50秒超时

        time.sleep(0.5)
        result_domain_tips = find_pic(img_domain_tips, area_domain_tips, 0.04)
        print("result_domain_tips", result_domain_tips)
        if result_domain_tips[0]:
            time.sleep(1)
            pa.click(1920 // 2, 800)
            time.sleep(0.1)
            return "True"
        
def find_blood_bar():
    while IS_FIND_BLOOD_BAR:
        img = screen_shot(321,193,1595-321,906-193 )
        # 筛选像素值
        # 找符合 (B=90, G=90, R=255) 的像素，生成掩码
        mask = (img[:, :, 0] == 90) & (img[:, :, 1] == 90) & (img[:, :, 2] == 255)
        mask = mask.astype(np.uint8)  # 变成 0和1 的图像（连通组件需要）

        # 获取连通组件的信息
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # 遍历所有连通块（注意：0是背景，不处理）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]  # 这个连通块的像素个数
            if area > 60:
                center_x, center_y = centroids[i]  # 中心点 (浮点数)
                print(f"连通块{i} - 面积：{area} - 中心坐标：(x={center_x:.2f}, y={center_y:.2f})")
                
                # 321,193
                diffrence = int(321 - center_x - 1920//2)
                print("diffrence:", diffrence)
                mouse_move_simulate(diffrence, 0) 
                time.sleep(0.5)

                continue   # 只要找一个
        
        print("no blood bar")
        mouse_move_simulate(600, 0)     # 没找到， 
        
        time.sleep(0.5)
        
def auto_fight_once():
    global IS_FIND_BLOOD_BAR
    IS_FIND_BLOOD_BAR = True
    threading.Thread(target=find_blood_bar).start()

    if AUTO_FIGHT_SCRIPT_KEY_LIST:
        for key, delay in AUTO_FIGHT_SCRIPT_KEY_LIST:
            kb.press("w")

            if key == "a":
                mouse.click(Button.left)
            elif key == "holde":
                kb.press("e")
                time.sleep(0.5)
                kb.release("e")
            elif key == "r":
                mouse.click(Button.right)
            else:
                pa.press(key)
            time.sleep(delay)

            kb.release("w")
    else:
        print("func (auto_fight_once): load AUTO_FIGHT_SCRIPT_KEY_LIST failed")
    
    IS_FIND_BLOOD_BAR = False
    time.sleep(1)

def line2angle(pos1, pos2, in_degrees=True):
    """
    计算 pos2 相对于 pos1 为原点时的极坐标角度。
    参数:
      pos1, pos2: (x, y) 二元组或列表
      in_degrees: 是否以度为单位返回（默认为 False，即返回弧度）
    返回:
      angle: 如果 in_degrees=False，返回弧度；否则返回 [0, 360) 度
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    # atan2 返回值在 [-π, π]
    angle = math.atan2(dy, dx)
    
    if in_degrees:
        angle = math.degrees(angle)  # 转为度
        # 将角度规范到 [0, 360)
        if angle < 0:
            angle += 360
    return angle

def get_reward():
    # 领取奖励, 转视角Y朝下
    mouse_move_simulate(0, 2000)
    time.sleep(0.5)
    while True:
        time.sleep(0.01)
        pa.press("f")

        result_reward = find_img(img_leylinereward_mat, region_leylinereward, 0.01)
        print("result_reward:", result_reward)
        if result_reward[0]:
            kb.release("w")
            time.sleep(0.2)
            print("领取奖励中...")
            time.sleep(2)

            # 领奖
            pa.click(747,694  )
            time.sleep(0.2)
            pa.click(853,696  )
            time.sleep(2)

            break

        # X朝向领奖处（宝箱图标）
        result_get_reward = find_img(leyline_completed_img_mat, (0, 0, 1920, 1080), 0.05)
        if result_get_reward[0]:
            x, y = result_get_reward[0], result_get_reward[1]
            angle = line2angle((1920//2, 1080//2), (x, y))
            print("angle:", angle)
            isFoward =  minimap_rotation2(angle, target_angle=270)
            if isFoward:
                kb.press("w")

    play_mp3(mp3_nomore_shuzi)  # DEBUG
    time.sleep(2)

def find_f_fight_get_reward():
    result_f = find_img(img_f_mat, area_f, 0.04)
    if result_f[0]:
        pa.press("f")
    else:
        return False

    while True:
        # fight
        auto_fight_once()

        # find reward
        result = find_img(leyline_completed_img_mat, (0, 0, 1920, 1080), 0.05)
        print(result)
        if result[0]:
            print("challege completed")
            break
        time.sleep(1)

    # get reward
    get_reward()


def find_ley_line_blossom(region, flag=None):
    result_money = find_img(img_money_mat, region, 0.05)
    print("result_money:", result_money)
    if result_money[0] and flag != "experience":
        return "money"
    
    result_experience = find_img(img_experience_mat, region, 0.05)
    print("result_experience", result_experience)
    if result_experience[0] and flag != "money":
        return "experience"
    
    return None

# “秋分山西则”传送点 右侧地脉之花
def func1(flag=None):
    # 传送到“临暴之城”， 放大到最大， 传送点 642,402  ，视角右转1205
    time.sleep(3)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_fendang)
    time.sleep(2)
    for i in range(8):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    # 传送副本1 “临暴之城”
    pa.click(pos_map_fendang_domain1)
    time.sleep(2)
    pa.click(pos_translate)
    finding_main_interface()  # 等待加载

    time.sleep(2)
    # 传送点
    pa.press("m")
    time.sleep(2)
    for i in range(6):
        pa.click(pos_map_zoom)
        time.sleep(0.2)

    # TODO 找图
    region1 = None  # “秋分山西则”传送点 右侧地脉之花region
    # return find_ley_line_blossom(img_money, region1)

# 枫丹左上部 money & experience
def func2(flag=None):
    """ one """
    time.sleep(3)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_fendang)
    time.sleep(2)
    for i in range(8):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    # 传送到 枫丹最上面七天神像
    pa.click(pos_map_fendan_top_qitianshenxiang)
    time.sleep(1)
    pa.click(pos_translate)
    time.sleep(2)
    finding_main_interface()
    time.sleep(2)
    pa.press("m")
    time.sleep(2)
    for i in range(3):  # 放大三次
        pa.click(pos_map_zoom)
        time.sleep(0.2)
    time.sleep(2)
    # 找图menoy 580,393,615,425   
    region = 580,393,615,425    # 
    result = find_ley_line_blossom(region, flag)
    if result:
        pa.click(624,435)   # 新枫丹科学院传送点 左侧传送点（水边）
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(1250, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(4.1)
        kb.press(Key.space)
        time.sleep(0.1)
        kb.release(Key.space)
        time.sleep(0.2)
        kb.press(Key.space)
        time.sleep(0.1)
        kb.release(Key.space)
        time.sleep(9.7)
        kb.release("w")

        find_f_fight_get_reward()

    
        """ two """
        time.sleep(3)
        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_mengDe)
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_fendang)
        time.sleep(2)
        for i in range(8):
            pa.click(pos_map_reduce)
            time.sleep(0.4)
        # 传送到 枫丹最上面七天神像
        pa.click(pos_map_fendan_top_qitianshenxiang)
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        time.sleep(2)
        pa.press("m")
        time.sleep(2)
        for i in range(3):  # 放大三次
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(2)

        pa.click(624,435)   # 新枫丹科学院传送点 左侧传送点（水边）
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(2350, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(30)
        kb.release("w")
        time.sleep(1)

        mouse_move_simulate(-700, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(5.8)
        kb.release("w")
        time.sleep(1)

        # 战斗
        find_f_fight_get_reward()

        """ three """
        time.sleep(3)
        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_mengDe)
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_fendang)
        time.sleep(2)
        for i in range(8):
            pa.click(pos_map_reduce)
            time.sleep(0.4)
        # 传送到 枫丹最上面七天神像
        pa.click(pos_map_fendan_top_qitianshenxiang)
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        time.sleep(2)
        pa.press("m")
        time.sleep(2)
        for i in range(3):  # 放大三次
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(2)

        pa.click(624,435)   # 新枫丹科学院传送点 左侧传送点（水边）
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_zoom)
        time.sleep(1)
        pa.click(854,355)   # 左上浪船锚点旁边传送点
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(-1210, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(12.6)
        kb.release("w")

        find_f_fight_get_reward()

        """ four """
        time.sleep(3)
        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_mengDe)
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_fendang)
        time.sleep(2)
        for i in range(8):
            pa.click(pos_map_reduce)
            time.sleep(0.4)

        # 传送到 枫丹最上面七天神像
        pa.click(pos_map_fendan_top_qitianshenxiang)
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        time.sleep(2)
        pa.press("m")
        time.sleep(2)
        for i in range(3):  # 放大三次
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(2)

        pa.click(624,435)   # 新枫丹科学院传送点 左侧传送点（水边）
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_zoom)
        time.sleep(1)
        pa.click(854,355)   # 左上浪船锚点旁边传送点
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(400, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(5)
        kb.release("w")
        time.sleep(1)

        mouse_move_simulate(152, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(8.8)
        kb.release("w")
        time.sleep(1)

        find_f_fight_get_reward()

""" 芒索斯山东麓 """
def func3(flag=None):
    """ one """
    time.sleep(3)
    pa.press("m")
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_mengDe)
    time.sleep(2)
    pa.click(pos_map_all)
    time.sleep(2)
    pa.click(pos_map_fendang)
    time.sleep(2)
    for i in range(8):
        pa.click(pos_map_reduce)
        time.sleep(0.4)
    time.sleep(1)
    pa.click(909,465)   # '深潮的余响' 副本
    time.sleep(1)
    pa.click(pos_translate)
    time.sleep(2)
    finding_main_interface()

    # 放大三次
    pa.press("m")
    time.sleep(2)
    for i in range(3):
        pa.click(pos_map_zoom)
        time.sleep(0.2)
    time.sleep(1)

    region = 801,305,839,339  
    result = find_ley_line_blossom(region, flag)
    if result:
        pa.click(808,279  ) # 芒索斯山东麓 传送锚点 
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(560, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(14.9)
        kb.release("w")
        time.sleep(1)

        find_f_fight_get_reward()

        """ two """
        time.sleep(3)
        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_mengDe)
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_fendang)
        time.sleep(2)
        for i in range(8):
            pa.click(pos_map_reduce)
            time.sleep(0.4)
        time.sleep(1)
        pa.click(909,465)   # '深潮的余响' 副本
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        # 放大三次
        pa.press("m")
        time.sleep(2)
        for i in range(3):
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(1)

        # 传送到水下传送锚点
        pa.click(929,268  )     # 芒索斯山东麓 传送锚点 右侧 水下传送锚点
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        pa.press("m")
        time.sleep(2)

        for i in range(3):
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(1)

        pa.click(593,595)     # 芒索斯山东麓 传送锚点 
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(-280, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(7.9)
        kb.release("w")
        time.sleep(1)

        find_f_fight_get_reward()

        """ three """
        time.sleep(3)
        pa.press("m")
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_mengDe)
        time.sleep(2)
        pa.click(pos_map_all)
        time.sleep(2)
        pa.click(pos_map_fendang)
        time.sleep(2)
        for i in range(8):
            pa.click(pos_map_reduce)
            time.sleep(0.4)
        time.sleep(1)
        pa.click(909,465)   # '深潮的余响' 副本
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        # 放大三次
        pa.press("m")
        time.sleep(2)
        for i in range(3):
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(1)

        # 传送到水下传送锚点
        pa.click(929,268  )     # 芒索斯山东麓 传送锚点 右侧 水下传送锚点
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        pa.press("m")
        time.sleep(2)

        for i in range(3):
            pa.click(pos_map_zoom)
            time.sleep(0.2)
        time.sleep(1)

        pa.click(593,595)     # 芒索斯山东麓 传送锚点 
        time.sleep(1)
        pa.click(pos_translate)
        time.sleep(2)
        finding_main_interface()

        mouse_move_simulate(-1860, 0)
        time.sleep(1)
        kb.press("w")
        time.sleep(18)
        kb.release("w")
        time.sleep(1)

        find_f_fight_get_reward()


def main_script():
    global ACCOUNT_INDEX
    isDone = None   # 是否切换账号 标志

    for i in range(3):  # 账号循环w
        if isDone:
            ACCOUNT_INDEX += 1
            show_statu_queue.put("切换账号\n")
            threads = []
            with ThreadPoolExecutor(max_workers=1) as t:
                threads.append(t.submit(switch_account))
            result = threads[-1].result()
            show_statu_queue.put(f":{result}")
            isDone = False

        for j in range(10):  # 传送循环
            if isDone:
                break  # 切换账号

            show_statu_queue.put("开始运行___\n传送组队...\n")
            threads = []
            with ThreadPoolExecutor(max_workers=1) as t:
                threads.append(t.submit(translate))
            result = threads[-1].result()
            show_statu_queue.put(f":{result}")

            for k in range(99):  # 副本循环
                # 等待秘境加载
                show_statu_queue.put("等待进入副本...")
                with ThreadPoolExecutor(max_workers=1) as t:
                    threads.append(t.submit(find_domain_tips))
                result = threads[-1].result()
                show_statu_queue.put(f":{result}")

                # walk to Feqeqeqeqeqeq
                show_statu_queue.put("walk to F...")
                with ThreadPoolExecutor(max_workers=1) as t:
                    threads.append(t.submit(walk_to_f))
                result = threads[-1].result()
                show_statu_queue.put(f":{result}")
                if result == "False":
                    break  # 重新传送

                show_statu_queue.put("开始战斗...")
                with ThreadPoolExecutor(max_workers=1) as t:
                    threads.append(t.submit(fight))
                result = threads[-1].result()
                show_statu_queue.put(f":{result}")
                if result == "False":
                    break  # 重新传送

                show_statu_queue.put("走到秘境中间...")
                print("put start yolo...")
                send_to_walk_queue.put(("start_yolo", None))

                flag, _ = send_to_main_queue.get()  # 阻塞等待 walk to center
                if flag == "pause_yolo":
                    print("pause_yolo...(main)\n")

                # walk to F
                show_statu_queue.put("walk to F...")
                with ThreadPoolExecutor(max_workers=1) as t:
                    threads.append(t.submit(walk_to_f))
                result = threads[-1].result()
                show_statu_queue.put(f":{result}")
                if result == "False":
                    break  # 重新传送

                show_statu_queue.put("继续挑战...")
                with ThreadPoolExecutor(max_workers=1) as t:
                    threads.append(t.submit(challenge_continue))
                result = threads[-1].result()
                print(result)
                show_statu_queue.put(f":{result}")
                if result == "False":
                    break  # 重新传送
                elif result == "done":
                    isDone = True
                    break


""" 角色调试时使用的是 瓦雷莎 """
def auto_world_fight(action):
    time.sleep(1)
    pa.click(1920//2, 1080//2)

    if action == "摩拉":
        flag = "money"
    elif action == "经验":
        flag = "experience"
    elif action == "随便":
        flag = None

    func_list = [func3, func2, func2, func2]

    for func in func_list:  # 循环找地脉之花
        result = func(flag)


class GetQueueQThread(QThread):
    get_tree_img_queue_signal = Signal(np.ndarray)
    get_minimap_img_queue_signal = Signal(np.ndarray)
    get_statu_queue_signal = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            try:
                img_tree = show_tree_img_queue.get(timeout=0.01)
                self.get_tree_img_queue_signal.emit(img_tree)
            except:
                pass

            try:
                img_minimap = show_minimap_img_queue.get(timeout=0.01)
                self.get_minimap_img_queue_signal.emit(img_minimap)
            except:
                pass

            try:
                statu = show_statu_queue.get(timeout=0.01)
                self.get_statu_queue_signal.emit(statu)
            except:
                pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建 QLabel 控件用于显示视频流
        self.label_tree = QLabel(self)
        self.label_tree.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_tree.setGeometry(10, 10, 300, 80)

        self.label_minimap = QLabel(self)
        self.label_minimap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_minimap.setGeometry(320, 10, 80, 80)

        self.button_start_script = QPushButton("自动匹配", self)
        self.button_start_script.setGeometry(450, 10, 100, 25)
        self.button_start_script.clicked.connect(self.start_script)

        self.button_start_auto_money = QPushButton("自动地脉", self)
        self.button_start_auto_money.setGeometry(450, 45, 100, 25)
        self.button_start_auto_money.clicked.connect(self.start_auto_money)
        # self.button_start_auto_money.setEnabled(False)


        self.label_statu = QLabel(self)
        self.label_statu.setGeometry(670, 10, 120, 50)

        self.combo_box_domain_name = QComboBox(self)
        self.combo_box_domain_name.addItems(["雷本", "水本", "火本", "下落本"])
        self.combo_box_domain_name.setCurrentIndex(3)
        self.combo_box_domain_name.setGeometry(560, 10, 100, 25)

        self.combo_box_money_experience = QComboBox(self)
        self.combo_box_money_experience.addItems(["摩拉", "经验", "随便"])
        self.combo_box_money_experience.setCurrentIndex(2)
        self.combo_box_money_experience.setGeometry(560, 45, 100, 25)

        self.label_tips_auto_fight = QLabel(self)
        self.label_tips_auto_fight.setGeometry(670, 45, 90, 25)
        self.label_tips_auto_fight.setText("出招表：")

        self.combo_box_auto_fight = QComboBox(self)
        self.combo_box_auto_fight.setGeometry(720, 45, 100, 25)
        self.txts = os.listdir(AUTO_FIGHT_SCRIPT_DIR)
        self.combo_box_auto_fight.addItems(self.txts)

        self.button_test_script = QPushButton("<测试出招表", self)
        self.button_test_script.setGeometry(830, 45, 100, 25)
        self.button_test_script.clicked.connect(self.test_script)


        # 初始化UI
        self.setWindowTitle("Genshin impact script")
        self.setGeometry(0, 0, 1000, 90)

        threading.Thread(target=walk_to_domain_center).start()

        self.get_queue_qthread = GetQueueQThread()
        self.get_queue_qthread.get_tree_img_queue_signal.connect(self.show_img_tree)
        self.get_queue_qthread.get_minimap_img_queue_signal.connect(self.show_img_minimap)
        self.get_queue_qthread.get_statu_queue_signal.connect(self.show_statu)
        self.get_queue_qthread.start()

    # 测试出招表
    def test_script(self):
        self.load_auto_fight_key_list()
        threading.Thread(target=auto_fight_once).start()

    def load_auto_fight_key_list(self):
        global AUTO_FIGHT_SCRIPT_KEY_LIST
        txt_path = os.path.join(AUTO_FIGHT_SCRIPT_DIR, self.combo_box_auto_fight.currentText())
        print(txt_path)
        # 加载文件到出招列表
        with open(txt_path, "r") as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines:
            if line:
                key, delay = line.strip().split(",")
                AUTO_FIGHT_SCRIPT_KEY_LIST.append([key, float(delay)])
        print("AUTO_FIGHT_SCRIPT_KEY_LIST:\n", AUTO_FIGHT_SCRIPT_KEY_LIST)

    def start_auto_money(self):
        self.load_auto_fight_key_list() # 加载出招表

        if self.button_start_auto_money.isEnabled():
            self.button_start_auto_money.setEnabled(False)
            if self.combo_box_money_experience.currentText() == "摩拉":
                threading.Thread(target=auto_world_fight, args=("摩拉",)).start()
                show_statu_queue.put("开始自动地脉...摩拉")
            if self.combo_box_money_experience.currentText() == "经验":
                threading.Thread(target=auto_world_fight, args=("经验",)).start()
                show_statu_queue.put("开始自动地脉...经验")
            if self.combo_box_money_experience.currentText() == "随便":
                threading.Thread(target=auto_world_fight, args=("随便",)).start()
                show_statu_queue.put("开始自动地脉...随便")

    def start_script(self):
        global DOMAIN
        if self.button_start_script.isEnabled():
            self.combo_box_domain_name.setEnabled(False)
            DOMAIN = self.combo_box_domain_name.currentText()
            print(DOMAIN)
            self.button_start_script.setEnabled(False)
            threading.Thread(target=main_script).start()

    def show_img_tree(self, img):
        if not img is None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape
            bytes_per_line = c * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap(qimg)
            pixmap = pixmap.scaled(self.label_tree.size(), Qt.KeepAspectRatio)  # 缩放
            self.label_tree.setPixmap(pixmap)

    def show_img_minimap(self, img):
        if not img is None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape
            bytes_per_line = c * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap(qimg)
            pixmap = pixmap.scaled(self.label_minimap.size(), Qt.KeepAspectRatio)  # 缩放
            self.label_minimap.setPixmap(pixmap)

    def show_statu(self, statu):
        self.label_statu.setText(f"{statu}")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
