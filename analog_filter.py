import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, cheby2, ellip, freqs

# 滤波器设计参数
order = 4  # 滤波器阶数
cutoff = 1000  # 截止频率（Hz）

# 设计 Butterworth 滤波器
b_butter, a_butter = butter(order, cutoff, btype='low', analog=True)

# 设计 Chebyshev Type I 滤波器
b_cheby1, a_cheby1 = cheby1(order, 1, cutoff, btype='low', analog=True)

# 设计 Chebyshev Type II 滤波器
b_cheby2, a_cheby2 = cheby2(order, 20, cutoff, btype='low', analog=True)

# 设计 Elliptic 滤波器
b_ellip, a_ellip = ellip(order, 1, 20, cutoff, btype='low', analog=True)

# 计算频率响应
w, h_butter = freqs(b_butter, a_butter, worN=np.logspace(1, 5, 1000))
w, h_cheby1 = freqs(b_cheby1, a_cheby1, worN=np.logspace(1, 5, 1000))
w, h_cheby2 = freqs(b_cheby2, a_cheby2, worN=np.logspace(1, 5, 1000))
w, h_ellip = freqs(b_ellip, a_ellip, worN=np.logspace(1, 5, 1000))

# 绘制频率响应
plt.figure(figsize=(12, 8))
plt.plot(w, 20 * np.log10(np.abs(h_butter)), label='Butterworth')
plt.plot(w, 20 * np.log10(np.abs(h_cheby1)), label='Chebyshev Type I')
plt.plot(w, 20 * np.log10(np.abs(h_cheby2)), label='Chebyshev Type II')
plt.plot(w, 20 * np.log10(np.abs(h_ellip)), label='Elliptic')
plt.xscale('log')
plt.title('Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()
plt.legend()

# 保存结果
plt.tight_layout()
plt.savefig('analog_filter.png')
plt.show()