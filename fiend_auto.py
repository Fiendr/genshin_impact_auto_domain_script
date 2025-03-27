
import pyautogui
import cv2
import time
import os
import re


'''区域截屏找图
template_img_path  模板图片, 
pos  左上 右下 坐标元组, 共四个参数
threshold为匹配阈值,0.04找不到, 0.0006能找到, 0.00000018非常匹配,建议设为0.001
返回一个元组: 
    (坐标X, 坐标Y, 域值)
'''
def find_pic(template_img_path, pos, threshold):
    screenshot = pyautogui.screenshot()
    cropped_img = screenshot.crop(pos)
    cropped_img.save('cropped_img.png')

    img = cv2.imread('cropped_img.png', cv2.IMREAD_GRAYSCALE)  # 转换为灰度图
    t_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图

    if img is None or t_img is None:
        print("图像加载失败，请检查路径和文件是否存在。")
        return None, None, None

    height, width = t_img.shape  # 获取模板图像的高度和宽度

    result = cv2.matchTemplate(img, t_img, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    t_pos = min_loc[0] + width / 2, min_loc[1] + height / 2

    if min_val > threshold:
        return None, None, min_val
    else:
        return t_pos[0] + pos[0], t_pos[1] + pos[1], min_val
    

if __name__ == '__main__':

    pass
