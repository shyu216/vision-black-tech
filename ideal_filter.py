import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread("demo.jpg", cv2.IMREAD_GRAYSCALE)

# 显示原始图像
plt.figure(figsize=(10, 10))
plt.subplot(421), plt.imshow(image, cmap='gray')
plt.title("Original Image"), plt.xticks([]), plt.yticks([])

# 对图像进行二维傅里叶变换
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)  # 将零频率分量移到中心

# 计算幅度谱并显示
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
plt.subplot(422), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])

# 获取图像的尺寸
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2  # 中心点

# 创建低频滤波器掩码（去除高频）
low_pass_mask = np.zeros((rows, cols), np.uint8)
r_outer = 60  # 外半径（高频截止）
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2
low_pass_mask[mask_area <= r_outer ** 2] = 1

# 创建高频滤波器掩码（去除低频）
high_pass_mask = np.ones((rows, cols), np.uint8)
r_inner = 60  # 内半径（低频截止）
high_pass_mask[mask_area <= r_inner ** 2] = 0

# 显示低频滤波器掩码
plt.subplot(423), plt.imshow(low_pass_mask, cmap='gray')
plt.title("Low-Pass Mask"), plt.xticks([]), plt.yticks([])

# 显示高频滤波器掩码
plt.subplot(424), plt.imshow(high_pass_mask, cmap='gray')
plt.title("High-Pass Mask"), plt.xticks([]), plt.yticks([])

# 将低频滤波器掩码应用到傅里叶变换结果上
f_transform_low_pass = f_transform_shifted * low_pass_mask
f_transform_low_pass_shifted = np.fft.ifftshift(f_transform_low_pass)
image_low_pass = np.fft.ifft2(f_transform_low_pass_shifted)
image_low_pass = np.abs(image_low_pass)

# 将高频滤波器掩码应用到傅里叶变换结果上
f_transform_high_pass = f_transform_shifted * high_pass_mask
f_transform_high_pass_shifted = np.fft.ifftshift(f_transform_high_pass)
image_high_pass = np.fft.ifft2(f_transform_high_pass_shifted)
image_high_pass = np.abs(image_high_pass)

# 显示低频滤波后的傅里叶变换结果
plt.subplot(425), plt.imshow(np.log1p(np.abs(f_transform_low_pass)), cmap='gray')
plt.title("Low-Pass Filtered Spectrum"), plt.xticks([]), plt.yticks([])

# 显示高频滤波后的傅里叶变换结果
plt.subplot(426), plt.imshow(np.log1p(np.abs(f_transform_high_pass)), cmap='gray')
plt.title("High-Pass Filtered Spectrum"), plt.xticks([]), plt.yticks([])

# 显示低频滤波后的图像
plt.subplot(427), plt.imshow(image_low_pass, cmap='gray')
plt.title("Low-Pass Filtered Image"), plt.xticks([]), plt.yticks([])

# 显示高频滤波后的图像
plt.subplot(428), plt.imshow(image_high_pass, cmap='gray')
plt.title("High-Pass Filtered Image"), plt.xticks([]), plt.yticks([])

# plt.show() 
# 保存结果
plt.savefig('ideal_filter.png')
plt.close()