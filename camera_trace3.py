import cv2
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

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

alpha = 10
lambda_c = 16
r1 = 0.5
r2 = 0.05
chromAttenuation = 0.1
nlevels = 4

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化
ret, frame = cap.read()
(h, w) = frame.shape[:2]
frame = cv2.resize(frame, (w//2, h//2))
if not ret:
    print("无法接收帧 (stream end?). Exiting ...")
    cap.release()
    exit()

# 手动选择三个 ROI
rois = []
for i in range(3):
    roi = cv2.selectROI(f'Select ROI {i+1}', frame, fromCenter=False, showCrosshair=True)
    rois.append(roi)
    cv2.destroyWindow(f'Select ROI {i+1}')

# 直接转换为YCrCb并归一化
frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
frame_y = frame_ycrcb[:, :, 0]
pyr = build_laplacian_pyramid(frame_y, levels=nlevels)
lowpass1 = copy.deepcopy(pyr)
lowpass2 = copy.deepcopy(pyr)

# 初始化 matplotlib 图形
fig, ax = plt.subplots()
intensity_queues = [deque(maxlen=100) for _ in range(3)]
lines = [ax.plot([], [], lw=2, label=f'ROI {i+1}')[0] for i in range(3)]
ax.set_xlim(0, 100)
ax.set_ylim(-0.1, 0.1)
ax.set_xlabel('Frame')
ax.set_ylabel('Average Intensity')
ax.legend()

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w//2, h//2))
    if not ret:
        return lines

    start_time = time.time()

    # 颜色空间转换和归一化合并处理
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    frame_y = frame_ycrcb[:, :, 0]
    pyr = build_laplacian_pyramid(frame_y, levels=nlevels)

    # 时域滤波（原地操作优化）
    for i in range(nlevels):
        cv2.addWeighted(lowpass1[i], 1 - r1, pyr[i], r1, 0, dst=lowpass1[i])
        cv2.addWeighted(lowpass2[i], 1 - r2, pyr[i], r2, 0, dst=lowpass2[i])
    
    filtered = [lp1 - lp2 for lp1, lp2 in zip(lowpass1, lowpass2)]

    # 空间频率放大（预计算参数优化）
    delta = lambda_c / 8 / (1 + alpha)
    exaggeration_factor = 2
    lambda_ = (frame.shape[0]**2 + frame.shape[1]**2)**0.5 / 3

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
    frame_ycrcb[:, :, 0] = upsampled
    output = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)

    # 计算每个 ROI 的平均强度
    roi_frame = upsampled
    for i, roi in enumerate(rois):
        roi_intensity = np.mean(roi_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
        intensity_queues[i].append(roi_intensity)

    # 更新 matplotlib 图形
    for i, line in enumerate(lines):
        line.set_data(range(len(intensity_queues[i])), intensity_queues[i])
    ax.set_xlim(0, max(len(intensity_queues[0]), 100))
    ax.set_ylim(-0.1, 0.1)

    print(f"延迟: {(time.time() - start_time)*1000:.2f}ms")
    cv2.putText(output, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for i, roi in enumerate(rois):
        roi_intensity = intensity_queues[i][-1]
        cv2.putText(output, f"ROI {i+1} Intensity: {roi_intensity:.2f}", (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(output, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
        cv2.putText(output, f"ROI {i+1}", (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Amplified Camera', output)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # 手动选择 ROI
        for i in range(3):
            roi = cv2.selectROI(f'Select ROI {i+1}', frame, fromCenter=False, showCrosshair=True)
            rois[i] = roi
            cv2.destroyWindow(f'Select ROI {i+1}')
    elif key == ord('q'):
        plt.close(fig)
        cap.release()
        cv2.destroyAllWindows()
        return lines

    return lines

# 使用 FuncAnimation 实现实时更新
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show()