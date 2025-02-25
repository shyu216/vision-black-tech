import numpy as np
import matplotlib.pyplot as plt

def plot_rgb_signal_demo():
    # 定义一个像素的RGB值随时间变化的信号
    time = np.linspace(0, 1, 500)
    r_signal = 0.5 * np.sin(2 * np.pi * 5 * time) + 0.5
    g_signal = 0.5 * np.sin(2 * np.pi * 10 * time) + 0.5
    b_signal = 0.5 * np.sin(2 * np.pi * 15 * time) + 0.5

    # 计算每个通道的傅里叶变换
    f_r = np.fft.fft(r_signal)
    f_g = np.fft.fft(g_signal)
    f_b = np.fft.fft(b_signal)

    # 计算频率轴
    freq = np.fft.fftfreq(len(time), d=(time[1] - time[0]))

    # 绘制RGB信号随时间变化的图像
    plt.figure(figsize=(12, 8))

    plt.subplot(321)
    plt.plot(time, r_signal, 'r')
    plt.title('Red Channel Signal (Time Domain)')
    plt.xlabel('Time')
    plt.ylabel('Intensity')

    plt.subplot(323)
    plt.plot(time, g_signal, 'g')
    plt.title('Green Channel Signal (Time Domain)')
    plt.xlabel('Time')
    plt.ylabel('Intensity')

    plt.subplot(325)
    plt.plot(time, b_signal, 'b')
    plt.title('Blue Channel Signal (Time Domain)')
    plt.xlabel('Time')
    plt.ylabel('Intensity')

    # 绘制RGB信号的频谱
    plt.subplot(322)
    plt.plot(freq, np.abs(f_r), 'r')
    plt.title('Red Channel Spectrum (Frequency Domain)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.subplot(324)
    plt.plot(freq, np.abs(f_g), 'g')
    plt.title('Green Channel Spectrum (Frequency Domain)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.subplot(326)
    plt.plot(freq, np.abs(f_b), 'b')
    plt.title('Blue Channel Spectrum (Frequency Domain)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_rgb_signal_demo()