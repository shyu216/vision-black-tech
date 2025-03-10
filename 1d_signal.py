import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 生成时间序列
sampling_rate = 1000  # 采样率
T = 1.0 / sampling_rate  # 采样间隔
N = 1000  # 采样点数
x = np.linspace(0.0, N*T, N, endpoint=False)  # 时间序列

# 生成复杂的1D信号（多个正弦波的叠加）
frequencies = [5, 50, 120]  # 信号频率
amplitudes = [1, 0.5, 0.2]  # 信号幅度
signal = np.zeros_like(x)
for frequency, amplitude in zip(frequencies, amplitudes):
    signal += amplitude * np.sin(2 * np.pi * frequency * x)

# 计算信号的傅里叶变换
yf = fft(signal)
xf = fftfreq(N, T)[:N//2]

# 绘制信号在空间域、时间域和频率域中的表现
fig, axs = plt.subplots(2, 1)

# 空间域/时间域，例如，沿一条直线的光强变化或心电图随时间的变化
axs[0].plot(x, signal)
axs[0].set_title('Spatial/Time Domain')
axs[0].set_xlabel('Position/Time')
axs[0].set_ylabel('Amplitude/Intensity')

# 频率域
axs[1].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
axs[1].set_title('Frequency Domain')
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Amplitude')

# 调整布局
plt.tight_layout()
plt.savefig('1d_signal.png')
plt.show()