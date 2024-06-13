import numpy as np
import matplotlib.pyplot as plt

# 读取npz文件
data = np.load('Bayesian_multiple_experiments.npz')
z_decode_gop = data['z_decode_gop']
z_decode_both = data['z_decode_both']
z_decode_mot = data['z_decode_mot']
z_decode_env = data['z_decode_env']
monte = 10

# 计算误差
error_gop = np.zeros([monte,])
error_both = np.zeros([monte,])
error_mot = np.zeros([monte,])
error_env = np.zeros([monte,])

for n in range(monte):
    error_gop[n] = np.std(z_decode_gop[n,:])
    error_both[n] = np.std(z_decode_both[n,:])
    error_mot[n] = np.std(z_decode_mot[n,:])
    error_env[n] = np.std(z_decode_env[n,:])

plt.figure(figsize=(8,5))
plt.plot(np.sqrt(error_both), label='Network Integration', linewidth=3)
plt.plot(np.sqrt(error_gop), label='Bayesian Integration', linewidth=3)
plt.xlabel('Trail Number', fontsize=22)
plt.ylabel('Mean Square Error', fontsize=22)
plt.legend(fontsize=12)
plt.tight_layout()

# 计算均值和标准差
mean_env = np.mean(error_env)
mean_gop = np.mean(error_gop) 
mean_both = np.mean(error_both) 
mean_mot = np.mean(error_mot)

std_env = np.std(error_env)
std_gop = np.std(error_gop)
std_both = np.std(error_both)
std_mot = np.std(error_mot)

# 绘制柱状图和误差条
labels = ['Only Env', 'Only Motion', 'Integration', 'Bayesian']
means = [mean_env, mean_mot, mean_both, mean_gop]
stds = [std_env, std_mot, std_both, std_gop]

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(6,5))
ax.bar(x, means, yerr=stds, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.set_ylabel('Mean Square Error', fontsize=22)
ax.tick_params(axis='y', labelsize=22)
plt.tight_layout()
# ax.set_title('Mean and Standard Deviation of Errors')

plt.show()
