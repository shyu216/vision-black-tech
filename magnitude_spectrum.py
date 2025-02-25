import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_magnitude_spectrum_rgb(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 分离RGB通道
    r, g, b = cv2.split(img_rgb)

    # 计算每个通道的2D傅里叶变换
    f_r = np.fft.fft2(r)
    f_g = np.fft.fft2(g)
    f_b = np.fft.fft2(b)

    # 将零频率分量移到中心
    fshift_r = np.fft.fftshift(f_r)
    fshift_g = np.fft.fftshift(f_g)
    fshift_b = np.fft.fftshift(f_b)

    # 计算幅度谱
    magnitude_spectrum_r = 20 * np.log(np.abs(fshift_r))
    magnitude_spectrum_g = 20 * np.log(np.abs(fshift_g))
    magnitude_spectrum_b = 20 * np.log(np.abs(fshift_b))

    # 绘制原始图像和幅度谱
    plt.figure(figsize=(24, 6))

    plt.subplot(141)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(magnitude_spectrum_r, cmap='gray')
    plt.title('Magnitude Spectrum - Red Channel')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')

    plt.subplot(143)
    plt.imshow(magnitude_spectrum_g, cmap='gray')
    plt.title('Magnitude Spectrum - Green Channel')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')

    plt.subplot(144)
    plt.imshow(magnitude_spectrum_b, cmap='gray')
    plt.title('Magnitude Spectrum - Blue Channel')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')

    # 添加一个颜色条
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(cmap='gray'), cax=cbar_ax, label='Magnitude')

    plt.show()

if __name__ == "__main__":
    plot_magnitude_spectrum_rgb('demo.jpg')