import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取npz文件
data = np.load('Bayesian_multiple_experiments.npz')
z_decode_gop = data['z_decode_gop']
z_decode_both = data['z_decode_both']
z_decode_mot = data['z_decode_mot']
z_decode_env = data['z_decode_env']
monte = 10

# 将矩阵转换为向量
vector_gop = z_decode_gop.flatten()
vector_both = z_decode_both.flatten()
vector_mot = z_decode_mot.flatten()
vector_env = z_decode_env.flatten()

# 计算CSV数据的直方图
hist, bins = np.histogram(vector_gop, bins=10)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 定义高斯函数
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# 拟合高斯函数到数据
popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(vector_gop), np.std(vector_gop)])

# 绘制直方图
plt.figure(figsize=(8, 5))
# 绘制CSV数据的高斯拟合曲线
x_fit = np.linspace(min(bin_centers) - 0.05, max(bin_centers) + 0.05, 1000)
y_fit = gaussian(x_fit, *popt)
# plt.hist(vector_both-62, bins=10, alpha=0.5, label='Both', linewidth=2)

plt.hist(vector_mot-62, bins=10, alpha=0.5, label='Motion')
plt.hist(vector_env-62, bins=10, alpha=0.5, label='Environment')
plt.hist(vector_both-62, bins=10, alpha=0.5, label='Integration')
plt.plot(x_fit-62, y_fit, label='Theoretical Bayesian', color='#FF6961', linewidth=5)
plt.xlim([-0.15, 0.15])

# 设置图例、标题和标签
plt.legend(fontsize=16)
plt.xlabel('Estimated Position', fontsize=22)
plt.ylabel('Frequency', fontsize=22)
# plt.title('Distribution of Decoded Values', fontsize=22)
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.show()
