"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例， 其他相机数据读取可能需要对应修改。

改动：接入 ARX5 机械臂 API，实时读取末端位姿并保存。
保存格式：x,y,z,roll,pitch,yaw （角度为弧度）
按键说明：
  - h：采集一帧图像并保存对应的机械臂末端位姿
  - q：退出采集
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import threading

count = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 使路径相对于当前脚本目录，避免工作目录变化带来的问题
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(_THIS_DIR, "collect_data")
os.makedirs(image_save_path, exist_ok=True)

# 将内部包路径加入 sys.path，便于导入内部模块
_PKG_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "arx_x5_python"))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    import keyboard_control as kc  # arx_x5_python/arx_x5_python/keyboard_control.py
except Exception as e:
    raise ImportError(
        f"无法导入 keyboard_control，请确认已构建 API 并存在 .so：{e}"
    )


def data_collect():
    global count
    try:
        # 启动键盘控制（curses）线程，终端里控制机械臂姿态
        ctrl_thread = threading.Thread(target=lambda: kc.curses.wrapper(kc.keyboard_control), daemon=True)
        ctrl_thread.start()
        print("已启动键盘控制（终端里操作），在图像窗口按 h 采集，按 q 退出采集")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                               cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

            k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            if k == ord('q'):
                print("收到退出指令，结束采集...")
                break
            if k == ord('h'):  # 键盘按一下h, 保存当前照片和机械臂位姿（在图像窗口内按键）
                print(f"采集第{count}组数据...")

                # 从 ARX5 获取当前末端位姿 [x,y,z,roll,pitch,yaw]，弧度
                try:
                    xyzrpy = kc.single_arm.get_ee_pose_xyzrpy()
                    pose = xyzrpy.tolist()
                except Exception as e:
                    print(f"获取机械臂位姿失败：{e}")
                    continue

                print(f"机械臂pose: {pose}")

                poses_txt = os.path.join(image_save_path, 'poses.txt')
                with open(poses_txt, 'a+', encoding='utf-8') as f:
                    # 以逗号分隔保存一行
                    pose_ = [str(float(i)) for i in pose]
                    new_line = f"{','.join(pose_)}\n"
                    f.write(new_line)

                img_path = os.path.join(image_save_path, f"{count}.jpg")
                cv2.imwrite(img_path, color_image)
                print(f"已保存: {img_path}")
                count += 1
    finally:
        # 资源清理
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    data_collect()
