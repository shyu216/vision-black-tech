import cv2
import numpy as np

# 读取图像
image = cv2.imread('demo.jpg')

# 检查图像是否成功读取
if image is None:
    print("无法读取图像")
    exit()

# 将图像从 BGR 转换为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像从 RGB 转换为 LAB
image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

# 将图像从 LAB 转换回 RGB
image_rgb_converted = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

# 将图像从 RGB 转换回 BGR 以便保存
image_bgr_converted = cv2.cvtColor(image_rgb_converted, cv2.COLOR_RGB2BGR)

# 显示原始图像和转换后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Converted Image', image_bgr_converted)

# 等待用户按下 'q' 键退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()