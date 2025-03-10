import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def build_wavelet_pyramid(image, wavelet, levels):
    """
    构建小波金字塔
    :param image: 输入图像
    :param wavelet: 小波类型
    :param levels: 金字塔层数
    :return: 小波金字塔列表
    """
    coeffs = pywt.wavedec2(image, wavelet, level=levels)
    wavelet_pyramid = []
    for i in range(levels + 1):
        cA = coeffs[0] if i == 0 else coeffs[i][0]
        wavelet_pyramid.append(cA)
    return wavelet_pyramid, coeffs

def reconstruct_image_from_wavelet(coeffs, wavelet):
    """
    从小波金字塔重建图像
    :param coeffs: 小波金字塔系数
    :param wavelet: 小波类型
    :return: 重建的图像
    """
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return reconstructed_image

def display(original_image, wavelet_pyramid, reconstructed_image):
    """
    显示小波金字塔和重建图像
    :param wavelet_pyramid: 小波金字塔列表
    :param reconstructed_image: 重建的图像
    """
    plt.figure(figsize=(15, 10))
    
    # 显示小波金字塔
    for i, image in enumerate(wavelet_pyramid):
        plt.subplot(2, len(wavelet_pyramid), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Wavelet Level {i}')
        plt.axis('off')
    
    # 显示原始图像
    plt.subplot(2, 2, 3)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示重建图像
    plt.subplot(2, 2, 4)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Wavelet Reconstructed Image')
    plt.axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig('wavelet_pyramid.png')
    plt.close()

def main():
    # 读取图像并转换为灰度图像
    image = cv2.imread('demo.jpg', cv2.IMREAD_GRAYSCALE)
    wavelet = 'haar'  # 使用Haar小波
    levels = 4  # 设置金字塔层数

    # 构建小波金字塔
    wavelet_pyramid, coeffs = build_wavelet_pyramid(image, wavelet, levels)
    # 重建图像
    reconstructed_image = reconstruct_image_from_wavelet(coeffs, wavelet)

    # 显示小波金字塔和重建图像
    display(image, wavelet_pyramid, reconstructed_image)

if __name__ == "__main__":
    main()