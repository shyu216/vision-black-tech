import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_gaussian_pyramid(image, levels):
    """
    构建高斯金字塔
    :param image: 输入图像
    :param levels: 金字塔层数
    :return: 高斯金字塔列表
    """
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)  # 向下采样
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    """
    构建拉普拉斯金字塔
    :param gaussian_pyramid: 高斯金字塔列表
    :return: 拉普拉斯金字塔列表
    """
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)  # 向上采样
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)  # 计算拉普拉斯层
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)  # 归一化
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # 最后一层直接添加
    return laplacian_pyramid

def reconstruct_image(laplacian_pyramid):
    """
    从拉普拉斯金字塔重建图像
    :param laplacian_pyramid: 拉普拉斯金字塔列表
    :return: 重建的图像
    """
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, laplacian_pyramid[i])
    return image

def display_pyramids_and_reconstruction(gaussian_pyramid, laplacian_pyramid, reconstructed_image):
    """
    显示高斯金字塔、拉普拉斯金字塔和重建图像
    :param gaussian_pyramid: 高斯金字塔列表
    :param laplacian_pyramid: 拉普拉斯金字塔列表
    :param reconstructed_image: 重建的图像
    """
    plt.figure(figsize=(10, 10))
    
    # 显示高斯金字塔
    for i, image in enumerate(gaussian_pyramid):
        plt.subplot(3, len(gaussian_pyramid), i + 1)
        plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(f'Gaussian Level {i}')
        plt.axis('off')
    
    # 显示拉普拉斯金字塔
    for i, image in enumerate(laplacian_pyramid):
        plt.subplot(3, len(laplacian_pyramid), len(gaussian_pyramid) + i + 1)
        plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(f'Laplacian Level {i}')
        plt.axis('off')
    
    # 显示重建图像
    plt.subplot(3, 1, 3)
    plt.imshow(cv2.cvtColor(reconstructed_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Laplacian Reconstructed Image')
    plt.axis('off')
    
    # plt.show() 
    # 保存结果
    plt.savefig('laplacian_pyramid.png')
    plt.close()

def main():
    # 读取图像
    image = cv2.imread('demo.jpg')
    levels = 4  # 设置金字塔层数

    # 构建高斯金字塔
    gaussian_pyramid = build_gaussian_pyramid(image, levels)
    # 构建拉普拉斯金字塔
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    # 重建图像
    reconstructed_image = reconstruct_image(laplacian_pyramid)

    # 显示高斯金字塔、拉普拉斯金字塔和重建图像
    display_pyramids_and_reconstruction(gaussian_pyramid, laplacian_pyramid, reconstructed_image)

if __name__ == "__main__":
    main()