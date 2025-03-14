import cv2
import numpy as np
import copy
import time

def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.insert(0, image)
    return gaussian_pyramid

alpha = 10
lambda_c = 10
r1 = 0.5
r2 = 0.05
chromAttenuation = 0.1
nlevels = 8

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化
ret, frame = cap.read()
if not ret:
    print("无法接收帧 (stream end?). Exiting ...")
    cap.release()
    exit()

# 直接转换为YCrCb并归一化
frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
frame_y = frame_ycrcb[:, :, 0]
pyr = build_gaussian_pyramid(frame_y, levels=nlevels)
lowpass1 = copy.deepcopy(pyr)
lowpass2 = copy.deepcopy(pyr)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (stream end?). Exiting ...")
        break

    cv2.imshow('Original Camera', frame)

    start_time = time.time()

    # 颜色空间转换和归一化合并处理
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    frame_y = frame_ycrcb[:, :, 0]
    pyr = build_gaussian_pyramid(frame_y, levels=nlevels)

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
        print(f"Apply alpha: {currAlpha if currAlpha > alpha else alpha} to level {l} ({filtered[l].shape[1]}x{filtered[l].shape[0]})")
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
    frame_ycrcb[:, :, 0] = frame_y + upsampled
    output = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)

    # print(f"延迟: {(time.time() - start_time)*1000:.2f}ms")
    cv2.putText(output, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Amplified Camera', output)

    # 合成并转换颜色空间
    frame_ycrcb[:, :, 0] = upsampled
    displacement_output = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('Displacement', displacement_output)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()