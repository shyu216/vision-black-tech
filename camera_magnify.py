import cv2
import numpy as np
from scipy.signal import butter, filtfilt

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

alpha = 5
lambda_c = 16
r1 = 0.4
r2 = 0.05
chromAttenuation = 0.1

# 打开摄像头
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

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2YCrCb)
pyr_y = build_laplacian_pyramid(frame[:, :, 0], levels=3)
pyr_cr = build_laplacian_pyramid(frame[:, :, 1], levels=3)
pyr_cb = build_laplacian_pyramid(frame[:, :, 2], levels=3)
pyr = [np.stack([pyr_y[i], pyr_cr[i], pyr_cb[i]], axis=-1) for i in range(len(pyr_y))]
lowpass1 = [np.copy(p) for p in pyr]
lowpass2 = [np.copy(p) for p in pyr]

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (stream end?). Exiting ...")
        break

    # 显示原始帧
    cv2.imshow('Original Camera', frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    pyr_y = build_laplacian_pyramid(frame[:, :, 0], levels=3)
    pyr_cr = build_laplacian_pyramid(frame[:, :, 1], levels=3)
    pyr_cb = build_laplacian_pyramid(frame[:, :, 2], levels=3)
    pyr = [np.stack([pyr_y[i], pyr_cr[i], pyr_cb[i]], axis=-1) for i in range(len(pyr_y))]

    nLevels = len(pyr)

    # temporal filtering
    lowpass1 = [(1 - r1) * lp1 + r1 * p for lp1, p in zip(lowpass1, pyr)]
    lowpass2 = [(1 - r2) * lp2 + r2 * p for lp2, p in zip(lowpass2, pyr)]
    filtered = [lp1 - lp2 for lp1, lp2 in zip(lowpass1, lowpass2)]

    # amplify each spatial frequency bands
    delta = lambda_c / 8 / (1 + alpha)
    exaggeration_factor = 2
    lambda_ = (frame.shape[0] ** 2 + frame.shape[1] ** 2) ** 0.5 / 3

    for l in range(nLevels):
        currAlpha = lambda_ / delta / 8 - 1
        currAlpha = currAlpha * exaggeration_factor

        if l == nLevels - 1 or l == 0:
            filtered[l] = np.zeros_like(filtered[l])
        elif currAlpha > alpha:
            filtered[l] = alpha * filtered[l]
        else:
            filtered[l] = currAlpha * filtered[l]

        lambda_ /= 2

    # Render on the input video
    output = np.zeros_like(frame)
    for j in range(3):
        upsampled = filtered[0][:, :, j]
        for l in range(1, nLevels):
            upsampled = cv2.pyrUp(upsampled, dstsize=(filtered[l].shape[1], filtered[l].shape[0]))
            upsampled += filtered[l][:, :, j]
        output[:, :, j] = upsampled

    output[:, :, 1] *= chromAttenuation
    output[:, :, 2] *= chromAttenuation

    output = frame + output
    output = cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = np.clip(output, 0, 1)

    amplified_frame = (output * 255).astype(np.uint8)

    # 显示放大帧
    cv2.imshow('Amplified Camera', amplified_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()