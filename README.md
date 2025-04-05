(现已兼容windows/utuntu x11)
本程序全部使用硬编码(代码很烂,能用就行,呵), 识别小地图摄像机朝向借荐(抄)了better genshin impact: https://github.com/babalae/better-genshin-impact
(ps:不会C#, 特地学了一星期opencv还是得抄, 噗!)
识别古树的YOLO预训练模型为绝缘本,草本, 水本, 火本总共2800张手动标注,(只有这几个本效果好)
# .pt文件下载: (放在genshin_script_asset目录)
    链接: https://pan.baidu.com/s/1V5R3O3JgOwstNcM63wfWZw?pwd=wwp4 提取码: wwp4

使用opencv模板匹配找图, 识别小地图上视角朝向
使用yolov8识别古树
使用uinput模拟鼠标移动
优化了BG调整视角的逻辑(BG会在调整完视角的最后一刻鼠标会飘几个像素, 就这一下让前面的视角调整全部白做, 不知道后面他们优化了没有;
所以我先视角朝东,再识别古树离中心的距离, 再根据距离移动, 再视角朝东, 再移动; 最多三次就行, 最后再根据古树的距离移动视角对齐)
# 演示视频:
    https://www.bilibili.com/video/BV1ZbZMYMEAN/


    

# 前置准备:
    *    pip install opencv-python mss torch ultralytics numpy PySide6 pynput pyautogui pygame scipy
    *    ubuntu下额外安装  python-uinput 库
    
    #   ubuntu x11(wayland下mss库用不了)  
        (windows下需要下载安装罗技鼠标cpp接口,驱动下载地址:
        链接: https://pan.baidu.com/s/18fUvqcbynPfFknRHLwqsgA?pwd=d5k5 提取码: d5k5
        已在window11测试通过) 
    #   队伍只有一个角色(且只能是萝莉,少男,少女; 尽量选个奶妈吧, 战斗的时候还放EQ呢)
    #   没有新手任务导航提示
    #   原神窗口分辨率只能是1280x720 且默认启动位置
    #   需要开启"临瀑之城"副本作为猎人本的跳板
    #   下载上面给的.pt文件解压放在genshin_script_asset目录

# 测试硬件:
    #   cpu AMD 3500X               # 推理耗时 1000ms    cpu总耗时8-10秒 找古树总共推理7次
    #   gpu AMD RTX 6700XT rocm6.3  # 推理耗时 40ms    gpu总耗时2-3秒 找古树总共推理11次 (Nvidia显卡可自行安装CUDA)
    #   ubuntu 22.04.5  X11 / win11
    #   python 3.11










###########################################################################
以下内容为GPT生成:
###########################################################################
Reasoned about Genshin Impact自动化脚本

# 原神自动化脚本

本项目提供了一个用于原神自动化的脚本，该脚本结合了计算机视觉、机器学习（YOLO模型）以及模拟键盘/鼠标输入，实现了游戏内传送、匹配、副本战斗等自动化操作。脚本还内置了基于 PySide6 的简单 GUI，用于实时显示状态信息、迷你地图朝向和检测到的目标（如树木）。

> **免责声明：**  
> 本工具仅用于学习和研究目的。使用该脚本自动化操作原神可能违反游戏的服务条款，并有可能导致账号封禁。请在使用或修改此脚本之前，充分了解相关风险和法律责任。

## 功能概览

- **自动传送与副本导航：**  
  自动在游戏中传送、匹配组队，并引导进入副本。

- **基于YOLO的目标检测：**  
  利用自定义训练的YOLO模型（基于2800张手动标注图片，训练100个epochs）检测关键游戏元素（如树木），以调整角色的方向和位置。

- **模拟输入控制：**  
  通过 PyAutoGUI、pynput 及 uinput 模块，实现键盘和鼠标的高精度模拟输入。

- **实时GUI反馈：**  
  通过 GUI 显示实时检测到的树木图像、迷你地图画面以及状态信息，便于调试和监控脚本运行状态。

- **多线程与队列管理：**  
  使用 Python 的 threading、ThreadPoolExecutor 以及 Queue，实现多任务并行处理和非阻塞队列消息传递。

## 测试环境

- **操作系统：**  
  推荐使用 Ubuntu 22.04.5 (X11)（其他Linux系统可能需要做相应调整）。

- **硬件要求：**  
  - CPU: AMD 3500X（CPU推理时间约1000ms）  
  - GPU: AMD RTX 6700XT（ROCm 6.3，GPU推理时间约40ms）

- **Python版本：**  
  Python 3.8 或以上

## 依赖库

请确保安装以下Python包：

- `opencv-python`
- `mss`
- `torch`
- `ultralytics`（YOLO）
- `numpy`
- `PySide6`
- `python-uinput`
- `pynput`
- `pyautogui`
- `pygame`
- `scipy`

可以通过 pip 命令安装：

```bash
pip install opencv-python mss torch ultralytics numpy PySide6 python-uinput pynput pyautogui pygame scipy
安装与配置
克隆项目：


下载资源：

确保将以下必要资源放置在 genshin_script_asset 目录下：

自定义YOLO模型：原神绝缘草本水本火本2800张100epochs_best.pt

图片资源（如 pipeizhong.bmp、main_interface.bmp 等）

音频资源（如 nomorshuzi.mp3）

配置屏幕区域与坐标：

脚本中预设了游戏界面各部分的坐标（如 region_x、region_y 等），默认基于 1280x720 分辨率。根据实际游戏窗口位置和大小，调整相应的区域坐标和鼠标点击位置。

使用说明
运行脚本：

使用 Python 运行脚本：

bash
复制
编辑
python your_script_name.py
操作GUI：

启动后会弹出一个窗口，其中包含：

树木检测图像： 显示带有检测框和推理时间的YOLO检测结果。

迷你地图： 显示当前迷你地图及其计算出的旋转角度。

状态信息： 展示文本状态更新，便于了解脚本当前运行阶段。

点击 “开始” 按钮启动自动化流程。

自动化流程：

脚本首先执行传送流程，通过模拟按键和鼠标点击操作在游戏内进行传送、匹配组队及副本进入操作。

YOLO 模块持续检测关键目标（如树木），并调整角色视角以便更精确地定位。

脚本还包含战斗、匹配以及挑战继续等流程的自动化逻辑。

代码结构
主要流程函数：

translate(): 处理传送和匹配流程。

fight(): 自动化战斗并监控副本通关情况。

challenge_continue(): 处理战斗后是否继续挑战的判断及操作。

walk_to_domain_center(): 利用YOLO检测目标，实现向副本中心移动。

工具函数：

screen_shot(): 截取指定区域的屏幕图像。

compute_mini_map_angle(): 计算迷你地图当前的旋转角度。

draw_angle(): 在迷你地图图像上绘制指示角度的直线。

mouse_move_simulate() 和 keyboard_press_simulate(): 模拟低级别的输入事件。

GUI组件：

基于 PySide6 构建，包括：

MainWindow 类：主窗口及控件布局。

GetQueueQThread 类：从队列中异步读取数据并更新GUI。

常见问题
性能问题：
如果使用 CPU 推理速度较慢，请确保 GPU 已正确配置且支持 ROCm。使用GPU推理能显著提高检测速度。

输入模拟失败：
请检查系统权限，确保有权限进行键鼠模拟操作。有时可能需要以 root 权限运行或配置 udev 规则。

屏幕区域匹配不准：
如果脚本无法正确识别游戏界面元素，请根据实际游戏窗口调整预设的坐标和区域参数。

贡献
欢迎提出 issue 或提交 pull request 以改进脚本、修复 bug 或增加新功能。


Setup and Installation
Clone the Repository:

bash
复制
编辑
git clone https://github.com/yourusername/genshin-impact-automation.git
cd genshin-impact-automation
Download Assets:

Ensure that you have the required assets in the genshin_script_asset directory:

Custom YOLO model: 原神绝缘草本水本火本2800张100epochs_best.pt

Image assets (e.g., pipeizhong.bmp, main_interface.bmp, etc.)

Audio assets (e.g., nomorshuzi.mp3)

Configure Screen Regions and Positions:

The script uses pre-defined coordinates for game interface elements. Verify and adjust the region coordinates (e.g., region_x, region_y, etc.) according to your display resolution (default is based on 1280x720).

Usage
Run the Script:

Execute the script using Python:

bash
复制
编辑
python your_script_name.py
GUI Operation:

The GUI window will open showing:

Tree Image Feed: Displays YOLO-detected tree images with bounding boxes and inference time.

Minimap Feed: Shows the minimap and current calculated rotation angle.

Status Display: Displays text status updates.

Click the "开始" (Start) button to launch the automation sequence.

Automation Workflow:

The script begins with the navigation routine, sending simulated key presses and mouse movements to:

Teleport between locations.

Initiate domain matches.

Engage in combat and process domain completion.

The YOLO module is used to continuously adjust the player’s orientation based on detected in-game objects.

Code Structure
Main Script Functions:

translate(): Handles the teleportation and matching process.

fight(): Automates combat and monitors for domain completion.

challenge_continue(): Manages post-combat sequences, such as deciding to continue or exit a challenge.

walk_to_domain_center(): Uses YOLO for adjusting player direction based on tree detection.

Utility Functions:

screen_shot(): Captures a screenshot of specified game regions.

compute_mini_map_angle(): Calculates the current minimap rotation based on image processing.

draw_angle(): Draws the calculated angle on the minimap image.

mouse_move_simulate() and keyboard_press_simulate(): Simulate low-level input events.

GUI Components:

Built with PySide6, featuring:

MainWindow class for UI.

GetQueueQThread class for handling asynchronous updates from various queues.

Troubleshooting
Performance Issues:
Verify that your GPU is correctly set up with ROCm if using GPU inference. Otherwise, the CPU inference path may be significantly slower.

Input Simulation Failures:
Ensure you have the necessary permissions to simulate input events on your system. Running as root or with appropriate privileges may be required.

Screen Region Mismatches:
If the script is not detecting in-game elements correctly, adjust the coordinates defined at the top of the script to match your game window’s layout.

Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests for improvements, bug fixes, or additional features.

License
This project is licensed under the MIT License. See the LICENSE file for details.

pgsql
复制
编辑

This README provides an overview of the project, setup instructions, usage guidelines, and troubleshooting tips to help you get started with the script.
