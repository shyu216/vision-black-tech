import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取图像
image = cv2.imread('demo.jpg', cv2.IMREAD_GRAYSCALE)

# 计算傅里叶变换
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# 计算幅度谱
magnitude_spectrum = np.abs(f_transform_shifted)

# 找出权重最大的10个频率成分
indices = np.unravel_index(np.argsort(magnitude_spectrum.ravel())[-10:], magnitude_spectrum.shape)

# 创建一个空的复数数组，用于存储合成图像的傅里叶变换
f_transform_combined = np.zeros_like(f_transform_shifted, dtype=complex)

# 可视化原始图像和幅度谱
plt.figure(figsize=(10, 20))

# 原始图像
plt.subplot(8, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 幅度谱
plt.subplot(8, 2, 2)
plt.imshow(20 * np.log(magnitude_spectrum + 1), cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

# 生成仅包含一个频率成分的图像并合成
for i, (row, col) in enumerate(zip(*indices)):
    single_freq_mask = np.zeros_like(f_transform_shifted, dtype=complex)
    single_freq_mask[row, col] = f_transform_shifted[row, col]
    
    # 计算仅包含一个频率成分的图像
    f_transform_single_freq = np.fft.ifftshift(single_freq_mask)
    image_single_freq = np.fft.ifft2(f_transform_single_freq)
    image_single_freq = np.abs(image_single_freq)
    
    # 将单频图像的傅里叶变换累加到合成图像的傅里叶变换中
    f_transform_combined += single_freq_mask
    
    # 可视化仅包含一个频率成分的图像
    plt.subplot(8, 2, i + 3)
    plt.imshow(image_single_freq, cmap='gray')
    plt.title(f'Single Frequency {i + 1}')
    plt.axis('off')

# 计算合成图像
f_transform_combined_shifted = np.fft.ifftshift(f_transform_combined)
image_combined = np.fft.ifft2(f_transform_combined_shifted)
image_combined = np.abs(image_combined)

# 可视化合成图像
plt.subplot(8, 2, 13)
plt.imshow(image_combined, cmap='gray')
plt.title('Combined Image')
plt.axis('off')
   
# plt.show() 
# 保存结果
plt.savefig('fourier_transform.png')
plt.close()