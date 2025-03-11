import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.insert(0, image)
    return gaussian_pyramid

def build_laplacian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    laplacian_pyramid = []
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    
    return laplacian_pyramid

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

# 将深度图像归一化到 0-255 范围内以便显示
color_scale_factor = 255 / 10  # np.max(depth_image) # 假设目标深度范围小于10m
print(f"Color Scale Factor: {color_scale_factor}")
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=color_scale_factor), cv2.COLORMAP_JET)

# 手动选择 ROI
roi = cv2.selectROI('Select ROI', depth_colormap, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select ROI')

pyr = build_laplacian_pyramid(depth_image, levels=nlevels)
laplowpass1 = copy.deepcopy(pyr)
laplowpass2 = copy.deepcopy(pyr)
gaulowpass1 = copy.deepcopy(pyr)
gaulowpass2 = copy.deepcopy(pyr)

# 初始化 matplotlib 图形
fig, ax = plt.subplots()
depth_queues = [deque(maxlen=100) for _ in range(3)]
labels = ['Origin', 'Lap Pyr Amplified', 'Gau Pyr Amplified']
lines = [ax.plot([], [], lw=2, label=f'{label}')[0] for label in labels]
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.set_xlabel('Frame')
ax.set_ylabel('Average depth')
ax.legend()

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return lines

    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_image /= 1000.0  # 假设深度值以毫米为单位

    # 将深度图像归一化到 0-255 范围内以便显示
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=color_scale_factor), cv2.COLORMAP_JET)

    # 计算 ROI 的平均强度
    roi_frame = depth_image
    roi_depth = np.mean(roi_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    depth_queues[0].append(roi_depth)

    # 更新 matplotlib 图形
    lines[0].set_data(range(len(depth_queues[0])), depth_queues[0])
    ax.set_xlim(0, len(depth_queues[0]))
    ax.set_ylim(-0.1, 0.1)

    cv2.putText(depth_colormap, f"ROI Average Depth: {roi_depth:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示 ROI
    cv2.rectangle(depth_colormap, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    
    # 显示深度图像
    cv2.imshow('Depth Image', depth_colormap)

    start_time = time.time()

    pyr = build_laplacian_pyramid(depth_image, levels=nlevels)

    # 时域滤波（原地操作优化）
    for i in range(nlevels):
        cv2.addWeighted(laplowpass1[i], 1 - r1, pyr[i], r1, 0, dst=laplowpass1[i])
        cv2.addWeighted(laplowpass2[i], 1 - r2, pyr[i], r2, 0, dst=laplowpass2[i])
    
    filtered = [lp1 - lp2 for lp1, lp2 in zip(laplowpass1, laplowpass2)]

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

    # 计算拉普拉斯金字塔放大后的平均深度
    roi_depth_lap = np.mean(amplified_depth[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    depth_queues[1].append(roi_depth_lap)

    # 将深度图像归一化到 0-255 范围内以便显示
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(amplified_depth, alpha=color_scale_factor), cv2.COLORMAP_JET)

    print(f"延迟: {(time.time() - start_time)*1000:.2f}ms")
    cv2.putText(depth_colormap, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(depth_colormap, f"ROI Average Depth: {roi_depth:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示 ROI
    cv2.rectangle(depth_colormap, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    
    # 显示深度图像 
    cv2.imshow('Laplacian Pyramid Amplified Depth Image', depth_colormap)

    start_time = time.time()

    pyr = build_gaussian_pyramid(depth_image, levels=nlevels)

    # 时域滤波（原地操作优化）
    for i in range(nlevels):
        cv2.addWeighted(gaulowpass1[i], 1 - r1, pyr[i], r1, 0, dst=gaulowpass1[i])
        cv2.addWeighted(gaulowpass2[i], 1 - r2, pyr[i], r2, 0, dst=gaulowpass2[i])
    
    filtered = [lp1 - lp2 for lp1, lp2 in zip(gaulowpass1, gaulowpass2)]

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

    # 计算高斯金字塔放大后的平均深度
    roi_depth_gau = np.mean(amplified_depth[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    depth_queues[2].append(roi_depth_gau)

    # 将深度图像归一化到 0-255 范围内以便显示
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(amplified_depth, alpha=color_scale_factor), cv2.COLORMAP_JET)

    print(f"延迟: {(time.time() - start_time)*1000:.2f}ms")
    cv2.putText(depth_colormap, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(depth_colormap, f"ROI Average Depth: {roi_depth:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示 ROI
    cv2.rectangle(depth_colormap, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    
    # 显示深度图像
    cv2.imshow('Gaussian Pyramid Amplified Depth Image', depth_colormap)

    # 更新 matplotlib 图形
    lines[0].set_data(range(len(depth_queues[0])), depth_queues[0])
    lines[1].set_data(range(len(depth_queues[1])), depth_queues[1])
    lines[2].set_data(range(len(depth_queues[2])), depth_queues[2])
    ax.set_xlim(0, max(len(depth_queues[0]), len(depth_queues[1]), len(depth_queues[2])))
    ax.set_ylim(0, max(max(depth_queues[0]), max(depth_queues[1]), max(depth_queues[2])) + 1)

    if cv2.waitKey(1) == ord('q'):
        plt.close(fig)
        pipeline.stop()
        cv2.destroyAllWindows()
        return lines

    return lines

# 使用 FuncAnimation 实现实时更新
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show()