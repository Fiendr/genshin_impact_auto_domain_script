import threading
import mss
import torch
from ultralytics import YOLO
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QEventLoop
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, \
    QPushButton
# import uinput
import pynput
from pynput.mouse import Button
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
from queue import Queue
from lg_mouse_controller import *
# window 罗技驱动级鼠标模拟
M = MoveR()

show_tree_img_queue = Queue(1000)
show_minimap_img_queue = Queue(1000)
show_statu_queue = Queue(20)
send_to_main_queue = Queue(5)
send_to_walk_queue = Queue(5)

DOMAIN_NAME = ""

# 前置准备:
#   队伍只有一个角色
#   没有新手任务导航提示
#   窗口分辨率1280x720默认位置
#
# 测试硬件:
#   cpu AMD 3500X               # 推理耗时 1000ms
#   gpu AMD RTX 6700XT rocm6.3  # 推理耗时 40ms
#   ubuntu 22.04.5  X11



import pyautogui as pa
from fiend_auto import *
import pygame

pygame.init()
pygame.mixer.init()

thread_pool_executor = ThreadPoolExecutor(max_workers=10)

kb = pynput.keyboard.Controller()
mouse = pynput.mouse.Controller()
# # 驱动级键鼠模拟
# events = (
#     uinput.REL_X,
#     uinput.REL_Y,
#     uinput.BTN_LEFT,
#     uinput.BTN_MIDDLE,
#     uinput.BTN_RIGHT,
# )

yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
# YOLO init 原神'绝缘草本水本火本'2800张手动标注 100epochs
model_path = "genshin_script_asset/原神绝缘草本水本火本2800张100epochs_best.pt"
model = YOLO(model_path)  # 禁用详细日志m
model = model.to(yolo_device)

# 基于1280 x 720
region_x, region_y, region_width, region_height = 320, 192, 1280, 340
minimap_x, minimap_y, minimap_width, minimap_height = 361, 204, 142, 142
img_center_x = region_width // 2  # 相对于图片中心X坐标

# # 全局创建 uinput 设备
# device = uinput.Device(events, name="virtual-mouse2")

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
mp3_nomore_shuzi = r"genshin_script_asset/nomorshuzi.mp3"

"坐标"
pos_map_all = 1548, 869
pos_map_mengDe = 1287, 305
pos_map_daoQi = 1292, 376
pos_map_fendang = 1283, 445
pos_map_fendang_domain1 = 1106, 849
pos_map_fendang_domain2 = 753, 709
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
img_pipeizhong = r"genshin_script_asset/pipeizhong.bmp"
area_pipeizhong = 1146, 856, 1162, 882
img_pipeizhong_cancel = r"genshin_script_asset/pipeizhong_cancel.bmp"
area_pipeizhong_cancel = 807, 673, 930, 720
img_2p_nobody = r"genshin_script_asset/2P_nobody.bmp"
area_2p_nobody = 822, 518, 876, 577
# 派蒙
img_main_interface = r"genshin_script_asset/main_interface.bmp"
area_main_interface = 346, 210, 372, 236
# 组队界面 + 号
img_team_plus = r"genshin_script_asset/team_plus.bmp"
area_team_plus = 1099, 841, 1143, 896
# 放弃挑战 死亡
img_cancel_challenge = r"genshin_script_asset/cancel_challenge.bmp"
area_cancel_challenge = 706, 843, 732, 872
# N秒后自动退出 通关
img_auto_out = r"genshin_script_asset/completed.bmp"
area_auto_out = 923, 817, 1038, 843
# 退出秘境
img_out_domain = r"genshin_script_asset/out_domain.bmp"
area_out_domain = 704, 832, 899, 881
# 树脂耗尽
img_nomore_shuzi = r"genshin_script_asset/img_nomore_shuzi.bmp"
area_nomore_shuzi = 709, 682, 736, 713
# 副本首页tips  "点击任意位置关闭"
img_domain_tips = "genshin_script_asset/domain_tips.png"
area_domain_tips = 878, 641, 1024, 667
# F
img_f = "genshin_script_asset/f.png"
area_f = 1093 - 40, 641 - 101, 1117 - 40, 661 - 101


def finding_main_interface():
    for i in range(60):
        time.sleep(1)
        result_mian_interface = find_pic(img_main_interface, area_main_interface, 0.095)
        print("找派蒙:", result_mian_interface)
        if result_mian_interface[0]:
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

#
# def mouse_move_simulate(x, y, device):
#     # 模拟鼠标移动：
#     device.emit(uinput.REL_X, x)
#     device.emit(uinput.REL_Y, y)
#     print("mouse simulate x, y", x, y)

def mouse_move_simulate(x, y):
    M.move(int(x*1.5), int(y*1.5))


def keyboard_press_simulate(key, delay):
    kb.press(key)
    time.sleep(delay)
    kb.release(key)


#################################################################################################
def minimap_rotation():
    while True:
        """ 小地图朝向角度检测 """
        img_minimap = screen_shot(minimap_x, minimap_y, minimap_width, minimap_height)
        angle = compute_mini_map_angle(img_minimap)
        print("angle", angle)
        img_minimap = draw_angle(img_minimap, angle)
        show_minimap_img_queue.put(img_minimap)
        min_rotation = 2  # 最小旋转距离
        if 180 > angle > 2:
            diff_x = min(int(-((360 - angle) // 22)), -min_rotation)
            mouse_move_simulate(diff_x, 0)
            time.sleep(0.02)
            print("angle left")
        elif 357 > angle >= 180:
            diff_x = max(int((angle // 22)), min_rotation)
            mouse_move_simulate(diff_x, 0)
            time.sleep(0.02)
            print("angle right")
        elif angle <= 2 or angle >= 357:
            print("已朝向东方")
            break


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
                    difference, img = get_tree_difference()
                    if not difference:
                        continue  # 未检测到tree

                    show_tree_img_queue.put(img)
                    print("difference:", difference)
                    delay = abs(difference) / 200
                    if difference > 15:
                        kb.release("a")
                        time.sleep(0.1)
                        kb.press("d")
                        time.sleep(delay)
                        kb.release("d")
                    elif difference < -15:
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
                            difference, img = get_tree_difference()
                            show_tree_img_queue.put(img)
                            thread_pool_executor.submit(mouse_move_simulate, difference // 5, 0)
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


def draw_angle(img, angle):
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
        img = cv2.putText(img, str(angle), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return img


def screen_shot(x, y, width, height):
    with mss.mss() as sct:
        # monitor = {"top": 0, "left": 0, "width": 400, "height": 400}  用这个出错!得用下面括号里的
        screenshot = sct.grab({'left': x,
                               'top': y,
                               'width': width,
                               'height': height})
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
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
        result_f = find_pic(img_f, area_f, 0.04)
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


def main_script():
    global ACCOUNT_INDEX
    isDone = None   # 是否切换账号 标志

    for i in range(3):  # 账号循环w
        if isDone:
            ACCOUNT_INDEX += 1
            show_statu_queue.put("切换账号\n")
            threads_t = []
            with ThreadPoolExecutor(max_workers=1) as t:
                threads.append(t.submit(switch_account))
            result = threads_t[-1].result()
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

                # walk to F
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
        self.label_tree.setGeometry(10, 10, 410, 80)

        self.label_minimap = QLabel(self)
        self.label_minimap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_minimap.setGeometry(420, 10, 80, 80)

        self.label_statu = QLabel(self)
        self.label_statu.setGeometry(510, 40, 100, 50)

        self.button_start_script = QPushButton("开始", self)
        self.button_start_script.setGeometry(510, 10, 100, 25)
        self.button_start_script.clicked.connect(self.start_script)

        self.combo_box_domain_name = QComboBox(self)
        self.combo_box_domain_name.addItems(["雷本", "水本", "火本"])
        self.combo_box_domain_name.setCurrentIndex(2)
        self.combo_box_domain_name.setGeometry(620, 10, 100, 25)

        # 初始化UI
        self.setWindowTitle("Genshin impact script")
        self.setGeometry(0, 0, 800, 90)

        threading.Thread(target=walk_to_domain_center).start()

        self.get_queue_qthread = GetQueueQThread()
        self.get_queue_qthread.get_tree_img_queue_signal.connect(self.show_img_tree)
        self.get_queue_qthread.get_minimap_img_queue_signal.connect(self.show_img_minimap)
        self.get_queue_qthread.get_statu_queue_signal.connect(self.show_statu)
        self.get_queue_qthread.start()

    def start_script(self):
        global DOMAIN
        if self.button_start_script.isEnabled():
            self.combo_box_domain_name.setEnabled(False)
            DOMAIN = self.combo_box_domain_name.currentText()
            print(DOMAIN)
            self.button_start_script.setEnabled(False)
            threading.Thread(target=main_script).start()

    def show_img_tree(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        bytes_per_line = c * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap(qimg)
        pixmap = pixmap.scaled(self.label_tree.size(), Qt.KeepAspectRatio)  # 缩放
        self.label_tree.setPixmap(pixmap)

    def show_img_minimap(self, img):
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