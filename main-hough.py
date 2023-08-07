"""
@Name: main.py
@Auth: Huohuo
@Date: 2023/7/3-15:01
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
import time
from utils import detection_lane_hough, draw_line, show_line, open_close
# import detection_lane_hough

# 打开视频文件
# video_path = 'data/watercar.mp4'  # 视频文件路径
video_path = 'data/1.avi'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 定义视频编写器的输出文件名、编码器、帧率和分辨率等参数
output_file = 'data/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
fps = cap.get(cv2.CAP_PROP_FPS)  # 使用与输入视频相同的帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 使用与输入视频相同的宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 使用与输入视频相同的高度

# 创建视频编写器对象
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


# 检查视频文件是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print("视频帧率: {:.2f}".format(fps))

# 初始化计时器和帧计数器
start_time = time.time()
frame_count = 0

temp_line = [348, 168, 636, 219]
line_history = [[348, 168, 636, 219]]

# 读取并显示视频帧
while True:
    # 逐帧读取视频
    ret, frame = cap.read()
    # 将图像尺寸缩放
    # frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
    cv2.imshow("frame1", frame)


    # 如果成功读取帧
    if ret:
        # frame = cv2.imread('./data/test.jpg')
        image_resize = frame
        img_h, img_w = frame.shape[:2]

        # print(frame.shape[:2])

        # lines = [x0, y0, x1, y1]
        frame, line = detection_lane_hough.detection(frame, temp_line, line_history)
        temp_line = line
        # line_history.append(line)
        print("line_history", line_history)

        # 在帧上绘制帧率
        cv2.putText(frame, "FPS: {:.2f}".format(frame_count / (time.time() - start_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 保存视频
        out.write(frame)

        # 在窗口中显示帧
        cv2.imshow('Video Frame', frame)
        print("-----------frame-------------")
        # 帧计数器加一
        frame_count += 1

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # 视频帧读取完毕或发生错误时退出循环
        break

# 计算实际处理后的帧率
end_time = time.time()
processed_fps = frame_count / (end_time - start_time)
print("实时帧率: {:.2f}".format(processed_fps))

# 释放资源并关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()
