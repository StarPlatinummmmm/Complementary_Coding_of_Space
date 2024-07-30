import numpy as np
import matplotlib.pyplot as plt
import math

# 圆周距离函数

def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis[dis > np.pi] -= 2 * np.pi
    dis[dis < -np.pi] += 2 * np.pi
    return dis


# 初始化参数
N = 1000
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
lambda_1 = 7
lambda_2 = 11
Lambda = np.array([lambda_1, lambda_2])

z_truth = 30
sigma_phi = 0.05
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi + sigma_phi * np.random.randn(2)
ap = 0.8
ag = ap / Lambda * 2 * np.pi
sigma_g = 0.1
Ig = np.zeros((2, N))

for i in range(2):
    dis_theta = circ_dis(theta, phi_truth[i])
    fg = np.exp(-dis_theta**2 / (2 * (math.sqrt(2)*ag[i])**2))
    Ig[i, :] = fg + sigma_g * np.random.randn(N)

z_0 = 28
phi_0 = np.mod(z_0 / Lambda, 1) * 2 * np.pi

T = 2000
z_record = np.zeros(T)
phi_record = np.zeros((2, T))
phi_decode = phi_0
z_decode = z_0
dt = 0.001

for t in range(T):
    z_record[t] = z_decode
    phi_record[:, t] = phi_decode
    
    phi_z = np.mod(z_decode / Lambda, 1) * 2 * np.pi
    dis_phi = circ_dis(phi_z, phi_decode)
    
    for i in range(2):
        dis_theta = circ_dis(theta, phi_decode[i])
        
        fg = dis_theta * np.exp(-dis_theta**2 / (2 * (math.sqrt(2)*ag[i])**2)) / (math.sqrt(2) * ag[i])**2
        # partial ln P(rg|phi) / partial phi
        dphi_1 = np.sum(fg * Ig[i, :]) / N * 2 * np.pi / sigma_g**2
        # dphi_1 = np.sum(fg * Ig[i, :]) * (N / 2 * np.pi) / sigma_g**2

        # partial ln P(phi|z) / partial phi
        dphi_2 = 1 / sigma_phi**2 * dis_phi[i]
        dphi = dphi_1 + dphi_2

        phi_decode[i] = phi_decode[i] + dphi * dt
        if phi_decode[i] > 2 * np.pi:
            phi_decode[i] -= 2 * np.pi
        elif phi_decode[i] < 0:
            phi_decode[i] += 2 * np.pi
    
    # partial ln P(phi|z) / partial z
    dz = np.sum(2 * np.pi / (Lambda * sigma_phi**2) * dis_phi)
    z_decode -= dz * dt

fontsize = 20
linewidth = 2.5

plt.figure()
for i in range(2):
    plt.plot(theta, Ig[i, :], linewidth=linewidth)
plt.gca().tick_params(width=linewidth)
plt.gca().tick_params(labelsize=fontsize)
plt.savefig('figure1.png')

plt.figure()
plt.plot(z_record, 'b', linewidth=linewidth)
plt.plot([0, T], [z_truth, z_truth], 'r--', linewidth=linewidth)
plt.xlabel('T', fontsize=12)
plt.ylabel('location z', fontsize=fontsize)
plt.gca().tick_params(width=linewidth)
plt.gca().tick_params(labelsize=fontsize)
# plt.ylim([27.5, 30.5])
plt.savefig('figure2.png')

plt.figure()
plt.plot(phi_record[0, :], linewidth=linewidth)
plt.plot(phi_record[1, :], linewidth=linewidth)
plt.plot([0, T], [phi_truth[0], phi_truth[0]], '--', linewidth=linewidth)
plt.plot([0, T], [phi_truth[1], phi_truth[1]], '--', linewidth=linewidth)
plt.xlabel('T', fontsize=12)
plt.ylabel('Phase φ', fontsize=fontsize)
plt.gca().tick_params(width=linewidth)
plt.gca().tick_params(labelsize=fontsize)
plt.savefig('figure3.png')
plt.show()