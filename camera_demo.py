import cv2
import numpy as np

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

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (stream end?). Exiting ...")
        break

    # 构建拉普拉斯金字塔
    laplacian_pyramid = build_laplacian_pyramid(frame, 3)

    # 显示原始帧
    cv2.imshow('Original Camera', frame)

    # 显示拉普拉斯金字塔
    for i, laplacian in enumerate(laplacian_pyramid):
        cv2.imshow(f'Laplacian Pyramid {i}', laplacian)

    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()