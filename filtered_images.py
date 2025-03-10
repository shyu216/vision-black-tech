import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt
import cv2

# 滤波器设计参数
order = 4  # 滤波器阶数
cutoff = 0.1  # 截止频率（归一化频率，0.1 对应于 Nyquist 频率的 10%）

# 设计 Butterworth 滤波器
b_butter, a_butter = butter(order, cutoff, btype='low', analog=False)

# 设计 Chebyshev Type I 滤波器
b_cheby1, a_cheby1 = cheby1(order, 1, cutoff, btype='low', analog=False)

# 设计 Chebyshev Type II 滤波器
b_cheby2, a_cheby2 = cheby2(order, 20, cutoff, btype='low', analog=False)

# 设计 Elliptic 滤波器
b_ellip, a_ellip = ellip(order, 1, 20, cutoff, btype='low', analog=False)

# 读取图像并转换为灰度图像
image = cv2.imread('demo.jpg', cv2.IMREAD_GRAYSCALE)

# 对图像的每一行和每一列应用滤波器
def apply_filter(image, b, a):
    filtered_image = np.zeros_like(image)
    # 对每一行应用滤波器
    for i in range(image.shape[0]):
        filtered_image[i, :] = filtfilt(b, a, image[i, :])
    # 对每一列应用滤波器
    for j in range(image.shape[1]):
        filtered_image[:, j] = filtfilt(b, a, filtered_image[:, j])
    return filtered_image

# 应用不同的滤波器
image_butter = apply_filter(image, b_butter, a_butter)
image_cheby1 = apply_filter(image, b_cheby1, a_cheby1)
image_cheby2 = apply_filter(image, b_cheby2, a_cheby2)
image_ellip = apply_filter(image, b_ellip, a_ellip)

# 绘制原始图像和滤波后的图像
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(image_butter, cmap='gray')
plt.title('Butterworth Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(image_cheby1, cmap='gray')
plt.title('Chebyshev Type I Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(image_cheby2, cmap='gray')
plt.title('Chebyshev Type II Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(image_ellip, cmap='gray')
plt.title('Elliptic Filtered Image'), plt.xticks([]), plt.yticks([])

# 保存结果
plt.tight_layout()
plt.savefig('filtered_images.png')
plt.show()