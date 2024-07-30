import numpy as np
from scipy.spatial.distance import cdist
from functools import partial
from multiprocessing import Pool
import time

# 圆周距离函数
def circ_dis(theta1, theta2):
    return np.minimum(np.abs(theta1 - theta2), 2 * np.pi - np.abs(theta1 - theta2))

# 初始化参数
N = 50
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
lambda_1 = 7
lambda_2 = 11
Lambda = np.array([lambda_1, lambda_2])
L = lambda_1 * lambda_2
z_truth = 30
sigma_phi = 0.05
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi + sigma_phi * np.random.randn(2)
ap = 0.8
ag = ap / Lambda * 2 * np.pi
sigma_g = 0.1
Ig = np.zeros((2, N))

for i in range(2):
    dis_theta = circ_dis(theta, phi_truth[i])
    fg = np.exp(-dis_theta**2 / (2 * ag[i]**2))
    Ig[i, :] = fg + sigma_g * np.random.randn(N)

# 后验概率计算函数
def calculate_posterior(i, x, Lambda, theta, Ig, ag, sigma_phi, sigma_g, N):
    phi_x = np.mod(x[i] / Lambda, 1) * 2 * np.pi
    local_log_posterior = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            dis_theta_1 = circ_dis(theta, theta[j])
            fg_1 = np.exp(-dis_theta_1**2 / (2 * ag[0]**2))
            log_likelihood_1 = -np.sum((Ig[0, :] - fg_1)**2) / N * 2 * np.pi / sigma_g**2 / 2

            dis_theta_2 = circ_dis(theta, theta[k])
            fg_2 = np.exp(-dis_theta_2**2 / (2 * ag[1]**2))
            log_likelihood_2 = -np.sum((Ig[1, :] - fg_2)**2) / N * 2 * np.pi / sigma_g**2 / 2

            log_likelihood = log_likelihood_1 + log_likelihood_2
            log_prior = -1 / (2 * sigma_phi**2) * ((theta[j] - phi_x[0])**2 + (theta[k] - phi_x[1])**2)
            local_log_posterior[j, k] = log_likelihood + log_prior

    return i, local_log_posterior

# 初始化并行计算
if __name__ == '__main__':
    N_x = 500
    x = np.linspace(0, L, N_x)
    log_posterior = np.zeros((N_x, N, N))

    # 使用 partial 函数来固定参数
    partial_calculate_posterior = partial(calculate_posterior, x=x, Lambda=Lambda, theta=theta, Ig=Ig, ag=ag, sigma_phi=sigma_phi, sigma_g=sigma_g, N=N)

    start_time = time.time()
    with Pool() as pool:
        results = pool.map(partial_calculate_posterior, range(N_x))
    end_time = time.time()

    for i, local_log_posterior in results:
        log_posterior[i, :, :] = local_log_posterior

    # 找到矩阵中的最大值及其线性索引
    maxValue = np.max(log_posterior)
    linearIndex = np.argmax(log_posterior)
    I, J, K = np.unravel_index(linearIndex, log_posterior.shape)

    print(f'真实的 x 值: {z_truth}')
    print(f'对应的 x 值: {x[I]}')
    print(f'真实的 theta 值: ({phi_truth[0]}, {phi_truth[1]})')
    print(f'对应的 theta 值: ({theta[J]}, {theta[K]})')
    print(f'计算时间: {end_time - start_time:.2f} 秒')