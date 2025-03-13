import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import time

def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.insert(0, image)
    return gaussian_pyramid

alpha = 20
lambda_c = 16
r1 = 0.5
r2 = 0.05
chromAttenuation = 0.1
nlevels = 4

# 配置 RealSense 流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动流
pipeline.start(config)

try:
    # 初始化
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    if depth_image is None:
        print("无法接收帧 (stream end?). Exiting ...")
        pipeline.stop()
        exit()

    # 直接转换为浮点型并归一化
    depth_image = depth_image.astype(np.float32)
    depth_image /= 1000.0  # 假设深度值以毫米为单位

    pyr = build_gaussian_pyramid(depth_image, levels=nlevels)
    lowpass1 = copy.deepcopy(pyr)
    lowpass2 = copy.deepcopy(pyr)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_image /= 1000.0  # 假设深度值以毫米为单位

        # 将深度图像归一化到 0-255 范围内以便显示
        color_scale_factor = 255 / 10  # np.max(depth_image) # 假设目标深度范围小于10m
        print(f"Color Scale Factor: {color_scale_factor}")
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=color_scale_factor), cv2.COLORMAP_JET)

        # 显示深度图像
        cv2.imshow('Depth Image', depth_colormap)

        start_time = time.time()

        pyr = build_gaussian_pyramid(depth_image, levels=nlevels)

        # 时域滤波（原地操作优化）
        for i in range(nlevels):
            cv2.addWeighted(lowpass1[i], 1 - r1, pyr[i], r1, 0, dst=lowpass1[i])
            cv2.addWeighted(lowpass2[i], 1 - r2, pyr[i], r2, 0, dst=lowpass2[i])
        
        filtered = [lp1 - lp2 for lp1, lp2 in zip(lowpass1, lowpass2)]

        # 空间频率放大（预计算参数优化）
        delta = lambda_c / 8 / (1 + alpha)
        exaggeration_factor = 2
        lambda_ = (depth_image.shape[0]**2 + depth_image.shape[1]**2)**0.5 / 3

        for l in range(nlevels):
            if l == 0 or l == nlevels - 1:
                filtered[l].fill(0)
                continue
            
            currAlpha = (lambda_ / delta / 8 - 1) * exaggeration_factor
            if currAlpha > alpha:
                filtered[l] *= alpha
            else:
                filtered[l] *= currAlpha
            
            lambda_ /= 2

        # 金字塔重建
        upsampled = filtered[0].copy()
        for l in range(1, nlevels):
            upsampled = cv2.pyrUp(upsampled, dstsize=(filtered[l].shape[1], filtered[l].shape[0]))
            upsampled += filtered[l]

        # 合成并转换颜色空间
        amplified_depth = depth_image + upsampled

        # 将深度图像归一化到 0-255 范围内以便显示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(amplified_depth, alpha=color_scale_factor), cv2.COLORMAP_JET)

        # print(f"延迟: {(time.time() - start_time)*1000:.2f}ms")
        cv2.putText(depth_colormap, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Amplified Depth Image', depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # 停止流
    pipeline.stop()

cv2.destroyAllWindows()