import numpy as np
import matplotlib.pyplot as plt
from Network import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm

# 圆周距离函数
def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis[dis > np.pi] -= 2 * np.pi
    dis[dis < -np.pi] += 2 * np.pi
    return dis

# 初始化参数
num_p = 1000
num_g = 1000
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
lambda_1 = 7
lambda_2 = 11
Lambda = np.array([lambda_1, lambda_2])

z_truth = 30
sigma_phi = 0.05
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi + 0.00 * np.random.randn(2)
ap = 0.8
ag = ap / Lambda * 2 * np.pi
sigma_g = 0.1
Ig = np.zeros((2, num_g))
fg = np.zeros((2, num_g))

for i in range(2):
    dis_theta = circ_dis(theta, phi_truth[i])
    fg[i, :] = np.exp(-dis_theta**2 / (4 * ag[i]**2))
    Ig[i, :] = fg[i, :] + sigma_g * np.random.randn(num_g)


Lambda = np.array([7,11])
L = Lambda[0] * Lambda[1]
### Place cells
P_CANN = Place_net(z_min=0, z_max=L, num=num_p, a_p=ap)
### Grid cells
M = 2
G_CANNs = bm.NodeList()
for i in range(M):
    G_CANNs.append(Grid_net(z_min=0, z_max=L, num= num_g, num_hpc=num_p, L = Lambda[i], a_p = ap))
### Coupled Network
Coupled_model = Coupled_Net(HPC_model=P_CANN, MEC_model_list=G_CANNs, num_module=M)

def run_net(i, Ip, Ig): 
    Coupled_model.step_run(i, Ip, Ig)
    u_HPC = Coupled_model.HPC_model.u
    u_grid = Coupled_model.MEC_model_list[0].u
    I_mec = Coupled_model.I_mec
    phi_1 = Coupled_model.MEC_model_list[0].center
    phi_2 = Coupled_model.MEC_model_list[1].center
    phi_decode = bm.stack([phi_1, phi_2])
    z_decode = Coupled_model.HPC_model.center
    return u_HPC, u_grid, I_mec, z_decode, phi_decode

T = 500
indices = np.arange(T)
z0 = 28
phi_0 = np.mod(z0 / Lambda, 1) * 2 * np.pi
fg = np.zeros((2, num_g))
for i in range(2):
    dis_theta = circ_dis(theta, phi_0[i])
    fg[i, :] = np.exp(-dis_theta**2 / (4 * ag[i]**2))
x = np.linspace(0,L,num_p,endpoint=False)
dis_x = x-z0
fp = np.exp(-dis_x**2 / (4 * ag[i]**2))
I_place = np.repeat(fp[np.newaxis, :], T, axis=0)
I_grid = np.repeat(fg[np.newaxis, :, :], T, axis=0)
bm.for_loop(run_net, (indices, I_place, I_grid), progress_bar=True)


T = 5000
I_place = np.zeros([T, num_p])
I_grid = np.repeat(Ig[np.newaxis, :, :], T, axis=0)
indices = np.arange(T)
u_HPC, u_grid, I_mec, z_record, phi_record = bm.for_loop(run_net, (indices, I_place, I_grid), progress_bar=True)

print(z_record.shape)
print(phi_record.shape)
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
plt.ylim([27.5, 30.5])
plt.tight_layout()
plt.savefig('figure2.png')

plt.figure()
plt.plot(phi_record[:, 0]+np.pi, linewidth=linewidth)
plt.plot(phi_record[:, 1]+np.pi, linewidth=linewidth)
plt.plot([0, T], [phi_truth[0], phi_truth[0]], '--', linewidth=linewidth)
plt.plot([0, T], [phi_truth[1], phi_truth[1]], '--', linewidth=linewidth)
plt.xlabel('T', fontsize=12)
plt.ylabel('Phase φ', fontsize=fontsize)
plt.gca().tick_params(width=linewidth)
plt.gca().tick_params(labelsize=fontsize)
plt.savefig('figure3.png')
plt.show()
