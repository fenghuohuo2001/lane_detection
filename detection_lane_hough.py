"""
@Name: detection_lane_hough.py
@Auth: Huohuo
@Date: 2023/7/3-15:03
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
import math

# ------------------------------------------------------------
#                     图像光照增强
#   use_type = "gamma" :使用gamma变换进行彩色图像增强
#   use_type = "scale" :使用拉伸进行彩色图像增强
# ------------------------------------------------------------
def gamma_correction(image, gamma):
    # 创建一个空的Look-Up Table (LUT)
    lut = np.empty((1, 256), dtype=np.uint8)

    # 对每个像素值进行Gamma校正
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    # 应用Look-Up Table (LUT)进行Gamma校正
    corrected_image = cv2.LUT(image, lut)

    return corrected_image
# ------------------------------------------------------------
#           感兴趣区域提取
# point 为（x， y）组成的点集list。 从左下角开始，按照顺时针排列
#         以下实例中，取的是六边形区域 如下所示
#           13/32            21/32
#              3---------------4            4/9
#             /                  \
#         2 /        ROI区域       \ 5       9/18
#          |                        |
#          |                        |
#         1|________________________|6       7/8
# ------------------------------------------------------------
def get_roi_area(img):
    img_h, img_w = img.shape[0], img.shape[1]
    mask = np.zeros_like(img)
    points = [(0, img_h*7/8), (0, img_h*9/18), (img_w*11/32, img_h*4/9), (img_w*21/32, img_h*4/9), (img_w, img_h*9/18), (img_w, img_h*7/8)]
    region_of_interest_vertices = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, region_of_interest_vertices, 255)
    masked_edges =cv2.bitwise_and(img, mask)

    return masked_edges, points


# -------------------------------------------------------
#              使用透视变换将前景图转换为鸟瞰图
#
# -------------------------------------------------------
def Turn2BirdView(img, points):
    img_h = img.shape[0]
    img_w = img.shape[1]
    image_resize = img
    # 定义模板中的四角点坐标，从左上角开始的顺时针顺序，共四个角点坐标
    mask_points = np.array([[0, 0], [image_resize.shape[1], 0], [image_resize.shape[1], image_resize.shape[0]],
                            [0, image_resize.shape[0]]], np.float32)
    src_bird_points = np.array([[250, 169], [450, 166], [640, 360], [0, 360]], np.float32)
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_bird_points, mask_points)

    # 应用透视变换
    birdseye_image = cv2.warpPerspective(image_resize, perspective_matrix, (img_w, img_h))
    return birdseye_image
# -------------------------------------------------------
#                        sobel算子
# -------------------------------------------------------
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

# -------------------------------------------------------
#                   锐化滤波器
# -------------------------------------------------------
def sharpen(img):
    kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    # kernel_sharpen_2 = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]])

    output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
    output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
    return output_1

# -------------------------------------------------------
#             k-mean 图像分割（输入灰度图）
# -------------------------------------------------------
def kmean_seg(img, num_class=3):
    img_h, img_w = img.shape[0], img.shape[1]
    # 转一维
    pixel_data = img.reshape((img_h*img_w, 1))
    pixel_data = np.float32(pixel_data/255)
    # 定义聚类终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # 设置随机初始标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means聚类
    compactness, labels, centers = cv2.kmeans(pixel_data, num_class, None, criteria, 10, flags)
    # 生成最终图像
    dst = (labels.reshape((img_h, img_w))*255).astype(np.float32)

    return dst


# ---------------------------------------------------------
#           绘制车道线的汇聚矩形范围
#    （其实应该是一个点，但是为了增加鲁棒性，确定为圆形或矩形范围）
#    通过直线是否通过这个范围来筛选是否为所需车道线
# ---------------------------------------------------------
def LaneConverge(img):
    img_h, img_w = img.shape[0], img.shape[1]

    # 定义汇聚范围
    # 640*360 == 281 139 - 390 137 - 396 192 - 284 191
    #  w * h  == w*7/16 h*7/18 - w*5/8 h*7/18 - w*5/8 h*19/36 - w*7/16 h*19/36
    w_scale_left = 290 / 640
    w_scale_right = 390 / 640
    h_scale_top = 150 / 360
    h_scale_bottom = 180 / 360
    point_left_top = (int(img_w * w_scale_left), int(img_h * h_scale_top))
    point_right_top = (int(img_w * w_scale_right), int(img_h * h_scale_top))
    point_right_bottom = (int(img_w * w_scale_right), int(img_h * h_scale_bottom))
    point_left_bottom = (int(img_w * w_scale_left), int(img_h * h_scale_bottom))
    print(point_left_top, point_right_bottom)

    image_with_rectangle = cv2.rectangle(img, point_left_top, point_right_bottom, (0, 255, 0), 2)

    return image_with_rectangle

# ---------------------------------------------------------
#                   判断直线是否经过矩形框
# 1.获取矩形的边界信息，包括左上角坐标 (x1, y1)、右下角坐标 (x2, y2)。
# 2.根据直线的方程 y = mx + b，计算直线的斜率 m 和截距 b。
# 3.对于矩形的每条边，计算直线与边的交点。
# 4.检查交点是否在矩形的范围内，如果有任何一个交点在范围内，则直线与矩形相交。
# ---------------------------------------------------------
def Judge(img, line):
    '''
    :param lines: HoughlinesP()检测结果
    :return:
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    # 矩形框的边界信息
    # 定义汇聚范围
    # 640*360 == 281 139 - 390 137 - 396 192 - 284 191
    #  w * h  == w*7/16 h*7/18 - w*5/8 h*7/18 - w*5/8 h*19/36 - w*7/16 h*19/36
    w_scale_left = 290 / 640
    w_scale_right = 390 / 640
    h_scale_top = 150 / 360
    h_scale_bottom = 180 / 360
    point_left_top = (int(img_w * w_scale_left), int(img_h * h_scale_top))
    point_right_top = (int(img_w * w_scale_right), int(img_h * h_scale_top))
    point_right_bottom = (int(img_w * w_scale_right), int(img_h * h_scale_bottom))
    point_left_bottom = (int(img_w * w_scale_left), int(img_h * h_scale_bottom))


    # 直线信息
    line_start = (line[0], line[1])
    line_end = (line[2], line[3])
    k = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])   # 斜率
    b = line_start[1] - k * line_start[0]   # 截距

    # 判断直线是否与矩形的边相交
    # 矩形边的端点坐标
    x1, y1 = point_left_top
    x2, y2 = point_right_bottom
    # --------------------------------------
    # 计算直线与点point_left_top两条邻边的交点
    # --------------------------------------
    inter_x1 = (y1 - b) / k
    inter_y1 = k * x1 + b
    # 检查交点是否在边的范围内
    if min(x1, x2) <= inter_x1 <= max(x1, x2) or min(y1, y2) <= inter_y1 <= max(y1, y2):
        return True
    # --------------------------------------
    # 计算直线与点point_right_bottom两条邻边的交点
    # --------------------------------------
    inter_x2 = (y2 - b) / k
    inter_y2 = k * x2 + b
    if min(x1, x2) <= inter_x2 <= max(x1, x2) or min(y1, y2) <= inter_y2 <= max(y1, y2):
        return True

    return False

# -------------------------------------------------------
#                 可视化hough检测直线
# -------------------------------------------------------
def show_line_simple(lines, img, sigma=500):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)       # 斜率<0时， a<0
            b = np.sin(theta)

            k = b/a
            # 斜率范围约束
            if -7 < k < -3:
                print("斜率为：", k)
                x0 = int(a * rho - (sigma-100) * (-b))
                y0 = int(b * rho - (sigma-100) * (a))
                # bottom
                x1 = int(x0 - sigma*2 * (-b))
                y1 = int(y0 - sigma*2 * (a))
                print("x0, y0", x0, y0)
                # print("x1, y1, x2, y2", x1, y1, x2, y2)
                # cv2.circle(img, (x0, y0), 5, (0, 0, 255), -1)
                # cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
                # cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)
                print("x0, y0, x1, y1", x0, y0, x1, y1)
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            elif k > 0:
                x0 = int(a * rho)
                y0 = int(b * rho)
                x1 = int(x0 + sigma * (-b))
                y1 = int(y0 + sigma * (a))
                # print("x0, y0, x1, y1", x0, y0, x1, y1)
                # cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

# -------------------------------------------------------
#                 可视化hough检测端点连线
# -------------------------------------------------------
def show_line_start_end(lines, img):
    if lines is not None:
        count = 0
        lines_count = []               # # 若检测到多条直线，取斜率最小的
        k_count = []               # # 若检测到多条直线，取斜率最小的
        # for line in lines[0]:
        for i, line in enumerate(lines[:, 0, :]):
            # print(i)
            # ------------------------------
            #  透视变换，判断直线是否为车道线
            # ------------------------------
            if not Judge(img, line):
                # print("Judge i is", i)
                break
            # print("i is", i)
            # print(line)
            x0 = line[0]
            y0 = line[1]
            # cv2.circle(img, (x0, y0), 5, (0, 0, 255), -1)
            # bottom
            x1 = line[2]
            y1 = line[3]
            # cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
            if x0 == x1:
                # print("!!!!!!!!!!!")
                continue
            # print("line:", line)
            # -----------延长------------
            length = 1000    # 延长长度
            # 计算直线的方向向量
            dx = x1 - x0
            dy = y1 - y0
            # 直线相关参数
            k = dy/dx

            # 将符合条件的k筛选出
            # 这里的逻辑是 检测到第一个符合斜率条件的车道线就直接终止循环并返回
            if 0.10 < k < 0.50:
                print("count is ", count)
                print("斜率：", k)
                print("-----长度-----：", math.sqrt(dx**2 + dy ** 2))
                count = count + 1
                # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

                # 若检测结果中出现2条或以上直线 测试过程中 并未出现2条直线的情况
                # if count > 1:
                #     raise ValueError(print(f"Error !!!!!!!!!!!! 出现{count}条直线！！！！！！！！！！！！"))
                lines_count.append(line)    # 应该用索引

                k_count.append(k)

        # 返回斜率最小的
        return lines_count, k_count




# -------------------------------------------------------
#                      形态学运算
# open = True:      开运算
# open = False:     闭运算
# -------------------------------------------------------
def open_close(img, k_x=3, k_y=3, frequency=1, open=True):
    morph_kernel = np.ones((k_x, k_y), np.uint8)
    if open:
        img_erode = cv2.erode(img, morph_kernel, frequency)
        img_dilate = cv2.dilate(img_erode, morph_kernel, frequency)
        return img_dilate

    else:
        img_dilate = cv2.dilate(img, morph_kernel, frequency)
        img_erode = cv2.erode(img_dilate, morph_kernel, frequency)
        return img_erode

# ---------------------------------------------------------
#                      卡尔曼滤波
#    Xk = A * Xk-1 + B * uk + Wk
# 已知二维位置测量值，先验模型预测值为model_x,model_y
# 传感器测量值 --
# pre_lane 当前状态
# ---------------------------------------------------------
def kalman(pre_lane, current_lane, line_history):
    # -----------------------------------------------
    # 创建卡尔曼滤波器 状态转移矩阵维度=4 测量矩阵维度为2
    # -----------------------------------------------
    kalman_filter = cv2.KalmanFilter(dynamParams=4, measureParams=4)

    # 设置初始状态 (x, y, dx, dy)
    # 起点：348 168 终点：636 219
    x1, y1 = pre_lane[0], pre_lane[1]
    x2, y2 = pre_lane[2], pre_lane[3]
    dx1, dy1 = 0.1, 0.1
    dx2, dy2 = 0.1, 0.1
    v = [dx1, dy1, dx2, dy2]
    measure_value = [x1, y1, x2, y2, dx1, dy1, dx2, dy2]
    kalman_filter.statePre = np.array(measure_value, dtype=np.float32).reshape(8, 1)

    # paper '''A_Lane_Tracking_Method_Based_on_Progressive_Probabilistic_Hough_Transform''' (Bk = 0, uk = 0, t = 0.2)
    # 设置转移矩阵A（描述系统的动态变换） Xk-1是四维 A应该是4*4
    diag = 1
    t = 0.2
    kalman_filter.transitionMatrix = np.array([[diag, 0, 0, 0, t, 0, 0, 0],
                                               [0, diag, 0, 0, 0, t, 0, 0],
                                               [0, 0, diag, 0, 0, 0, t, 0],
                                               [0, 0, 0, diag, 0, 0, 0, t],
                                               [0, 0, 0, 0, diag, 0, 0, 0],
                                               [0, 0, 0, 0, 0, diag, 0, 0],
                                               [0, 0, 0, 0, 0, 0, diag, 0],
                                               [0, 0, 0, 0, 0, 0, 0, diag]
                                               ], dtype=np.float32).reshape(8, 8)


    # kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    #                                             [0, 1, 0, 0, 0, 0, 0, 0],
    #                                             [0, 0, 1, 0, 0, 0, 0, 0],
    #                                             [0, 0, 0, 1, 0, 0, 0, 0],
    #                                             [0, 0, 0, 0, 1, 0, 0, 0],
    #                                             [0, 0, 0, 0, 0, 1, 0, 0],
    #                                             [0, 0, 0, 0, 0, 0, 1, 0],
    #                                             [0, 0, 0, 0, 0, 0, 0, 1]
    #                                             ], dtype=np.float32).reshape(8, 8)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0]
                                                ], dtype=np.float32)
    # 设置过程噪声协方差矩阵
    kalman_filter.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1

    # 设置测量噪声协方差矩阵
    kalman_filter.measurementNoiseCov = np.eye(8, dtype=np.float32) * 1

    # 初始化测量值(前一帧的检测结果)
    measure_value = np.array(measure_value, dtype=np.float32).reshape(8, 1)

    # 车道线循环跟踪
    print("len(line_history)", len(line_history))
    if len(line_history) < 200:
        for i in range(len(line_history)):
            print("i", i)
            # 当前帧的检测结果
            current_lane = np.array(line_history[i]+v, dtype=np.float32).reshape(8, 1)
            print("current_lane", current_lane)

            # 使用前一帧的车道线检测结果更新卡尔曼滤波器状态
            kalman_filter.correct(current_lane)

            # 预测下一个时间步的位置
            prediction = kalman_filter.predict()
            print("prediction", prediction)
            # 获取估计的位置
            estimated_position = prediction[:8]
            # 使用当前帧的车道线检测结果矫正估计的位置
            # 假设当前帧检测到的车道线是 prev_lane 加上一些随机噪声
            current_lane_with_noise = measure_value + np.random.randn(8, 1) * 5

            # 使用矫正后的位置更新前一帧的车道线检测结果
            measure_value = estimated_position + (current_lane_with_noise - estimated_position) * 0.2
    # 取后200个元素
    else:
        for i in range(len(line_history[-200:])):
            print("i", i)
            # 当前帧的检测结果
            current_lane = np.array(line_history[-200:][i]+v, dtype=np.float32).reshape(8, 1)
            print("current_lane", current_lane)

            # 使用前一帧的车道线检测结果更新卡尔曼滤波器状态
            kalman_filter.correct(current_lane)

            # 预测下一个时间步的位置
            prediction = kalman_filter.predict()
            print("prediction", prediction)
            # 获取估计的位置
            estimated_position = prediction[:8]
            # 使用当前帧的车道线检测结果矫正估计的位置
            # 假设当前帧检测到的车道线是 prev_lane 加上一些随机噪声
            current_lane_with_noise = measure_value + np.random.randn(8, 1) * 5

            # 使用矫正后的位置更新前一帧的车道线检测结果
            measure_value = estimated_position + (current_lane_with_noise - estimated_position) * 0.2



    return measure_value[:4]

# -------------------------------------------------------------------------
#                    灰度化 + 边缘检测 + hough检测 主程序
# -------------------------------------------------------------------------
#  img： 输入图像
#  return：
# ----------以下为上一帧车道线检测结果 用于车道线保持--------
# left_mid_point ： 左中点
# right_mid_point ： 右中点
# slope_left_input： 左斜率
# slope_right_input：右斜率
# --------------------------------------------------------------------------
def detection(img, temp_line, line_history):
    # ------------------------------------------------------
    #                          预处理
    # ------------------------------------------------------
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray", img_gray)

    k = 9
    # img_blur_gas = cv2.GaussianBlur(src_gray, (k, k), 5)
    img_blur = cv2.bilateralFilter(src_gray, d=9, sigmaColor=100, sigmaSpace=15)
    # img_blur = cv2.bilateralFilter(img_gray, d=3, sigmaColor=100, sigmaSpace=15)
    # img_blur_medi = cv2.medianBlur(src_gray, 3, 0)
    # img_blur = cv2.blur(src_gray, (3, 3))
    # cv2.imshow("img_blur_gas", img_blur_gas)
    cv2.imshow("img_blur_bil", img_blur)
    # cv2.imshow("img_blur_medi", img_blur_medi)
    # cv2.imshow("img_blur", img_blur)

    # ------------------------------------------------------
    #                         锐化
    # ------------------------------------------------------
    img_sharpen = sharpen(img_blur)
    cv2.imshow("img_sharpen", img_sharpen)

    # ------------------------------------------------------
    #                      图像光照增强
    # ------------------------------------------------------
    img_en_gamma = gamma_correction(img_blur, gamma=2)
    cv2.imshow("img_en_gamma", img_en_gamma)


    # ------------------------------------------------------
    #                     转鸟瞰图
    # ------------------------------------------------------
    # img_bird = Turn2BirdView(img_cut, points)
    # cv2.imshow("img_bird", img_bird)

    # ------------------------------------------------------
    #                    分割
    # ---------------------------------------
    #      二值化---目前使用固定阈值效果较好
    # ---------------------------------------
    # img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    # _, img_binary = cv2.threshold(img_cut, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, img_binary = cv2.threshold(img_en_gamma, 120, 255, cv2.THRESH_BINARY)
    # _, img_binary = cv2.threshold(img_cut, 30, 255, cv2.THRESH_BINARY)
    # cv2.imshow("img_binary", img_binary)
    # ---------------------------------------
    #            kmean聚类分割
    # ---------------------------------------
    # img_kmeanseg = kmean_seg(img_en_gamma)
    # cv2.imshow("img_kmeanseg", img_kmeanseg)

    # ------------------------------------------------------
    #                        开运算
    # ------------------------------------------------------
    # img_open = open_close(img_binary, k_x=3, k_y=3, frequency=1, open=True)
    # cv2.imshow("img_open", img_open)

    # ------------------------------------------------------
    #                        边缘检测
    #  sobel < canny < laplacian
    # ------------------------------------------------------
    img_edge = cv2.Canny(img_en_gamma, 15, 45)
    # img_edge_50 = cv2.Canny(img_en_gamma, 10, 50)
    # img_edge_30 = cv2.Canny(img_en_gamma, 10, 30)
    # img_edge_45 = cv2.Canny(img_en_gamma, 15, 45)
    # cv2.imshow("img_edge_canny_50", img_edge_50)
    # cv2.imshow("img_edge_canny_30", img_edge_30)
    # cv2.imshow("img_edge_canny_45", img_edge_45)

    img_edge1 = cv2.Canny(img_sharpen, 10, 30)

    cv2.imshow("img_edge_canny", img_edge1)

    img_edge2 = sobel(img_en_gamma)
    img_edge3 = sobel(img_sharpen)
    _, img_edge4 = cv2.threshold(img_edge3, 60, 250, cv2.THRESH_BINARY)
    cv2.imshow("img_edge_sobel", img_edge2)
    cv2.imshow("img_edge_sobel3", img_edge3)
    cv2.imshow("img_edge_sobel4", img_edge4)

    # img_edge = cv2.Laplacian(img_open, cv2.CV_16S, ksize=3)
    # img_edge = cv2.convertScaleAbs(img_edge)
    cv2.imshow("img_edge_Lap", img_edge)


    # ------------------------------------------------------
    #    截取ROI区域   points = [(x0, y0), ....(xn, yn)]
    # ------------------------------------------------------
    img_cut, points = get_roi_area(img_edge)
    # img_cut= img_en_gamma
    cv2.imshow("img_cut", img_cut)


    # ------------------------------------------------------
    #                       hough检测
    # 返回的是每一条直线对应的（ρ，θ）
    # ρ的精度为1， θ精度为1°，长度阈值200
    # ------------------------------------------------------
    # lines = cv2.HoughLines(img_cut, 15, np.pi/20, 50)
    # show_line_simple(lines, img)

    # ------------------------------------------------------
    #                       hough端点检测
    # hough检测-返回端点
    # ------------------------------------------------------
    # lines = cv2.HoughLinesP(img_cut, rho=1, theta=np.pi/180, threshold=10,
    #                         minLineLength=150, maxLineGap=300)
    lines = cv2.HoughLinesP(img_cut, rho=1, theta=np.pi / 180, threshold=10,
                            minLineLength=150, maxLineGap=300)
    # lines = cv2.HoughLinesP(img_cut, rho=1, theta=np.pi/360, threshold=30)
    # lines = cv2.HoughLinesP(img_cut, rho=1, theta=np.pi/180, threshold=150)

    # print("lines", lines)
    edge_line = None
    result = show_line_start_end(lines, img)
    # edge_line, k_count = show_line_start_end(lines, img)
    if result != [] and result is not None:
        # print("result is", result)
        edge_line, k_count = result
        # print("edge_line is", edge_line)
        # print("k_count is", k_count)

    # print('temp_line', temp_line)
    # ------------------------------------------------------
    #       若当前帧检测到，则采用当前帧的检测结果
    # ------------------------------------------------------
    if edge_line is not None and edge_line != []:
        # print("edge_line", edge_line)
        # print("edge_line.shape", len(edge_line))
        # ----k越小，越竖直----
        # min_index = np.argmin(k_count)
        # edge_line = edge_line[min_index]
        # print("k min is", np.min(k_count))
        # ----k越大，越水平----
        max_index = np.argmax(k_count)
        edge_line = edge_line[max_index]
        # print("k max is", np.max(k_count))
        cv2.line(img, (edge_line[0], edge_line[1]), (edge_line[2], edge_line[3]), (0, 0, 255), 2)
        # 得到200条以上车道线后，再进行kalman滤波
        # kalman-filter
        kalman_result = kalman(temp_line, edge_line, line_history).reshape(4).astype(np.uint8)
        cv2.line(img, (kalman_result[0], kalman_result[1]), (kalman_result[2], kalman_result[3]), (0, 255, 0), 2)
    # ------------------------------------------------------
    #      若上一帧检测到，而下一帧没有检测到，则延续前一帧的检测结果
    # ------------------------------------------------------
    else:
        edge_line = temp_line
        if edge_line is not None:
            cv2.line(img, (edge_line[0], edge_line[1]), (edge_line[2], edge_line[3]), (0, 0, 255), 2)
        # kalman-filter
        kalman_result = kalman(temp_line, edge_line, line_history).reshape(4).astype(np.uint8)
        cv2.line(img, (kalman_result[0], kalman_result[1]), (kalman_result[2], kalman_result[3]), (0, 255, 0), 2)


    # ------------------------------------------------------
    #                  kalman滤波
    # ------------------------------------------------------
    # if edge_line is not None and temp_line is not None:
    #     # kalman-filter
    #     print("edge_line is:", edge_line)
    #     print("temp_line is:", temp_line)
    #     pre_lane = kalman(temp_line, edge_line)
    #
    #
    #     if pre_lane is not None:
    #         cv2.line(img, (pre_lane[0], pre_lane[1]), (pre_lane[2], pre_lane[3]), (0, 0, 255), 2)
    #         return img, pre_lane

    cv2.imshow("img_line", img)
    # cv2.waitKey(0)
    img = LaneConverge(img)
    return img, edge_line




if __name__ == "__main__":
    img_path = "../data/start.png"
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]
    print("img.shape is", image.shape)
    image = cv2.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
    img, img_edge, img_cut, left_mid_point_xs, right_mid_point_xs, slope_lefts, slope_rights, lane = detection(image)



    cv2.waitKey(0)
    cv2.destroyAllWindows()
