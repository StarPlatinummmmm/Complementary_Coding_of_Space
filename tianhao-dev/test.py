import numpy as np
import matplotlib.pyplot as plt
from Network import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import matplotlib as mpl

# Parameters
# grid spacing
lambda_1 = 3
lambda_2 = 7
lambda_3 = 10
Lambda = np.array([lambda_1, lambda_2, lambda_3])
Coding_range = lambda_1 * lambda_2 * lambda_3
# connection range
a_p = 0.3
a_g = a_p/Lambda*2*np.pi
# connection strength
J_p = 20
J_g = J_p
J_pg = J_p/50
# divisive normalization
k_p = 20.
k_g = Lambda/2/np.pi * k_p
# time constants
tau_p = 1.
tau_g = 2*np.pi * tau_p/Lambda
# cell number
num_p = int(1280)*2
rho_p = num_p/Coding_range
rho_g = rho_p
num_g = int(rho_g*2*np.pi) # 为了让两个网络的rho相等

# 生成Ip, Ig
sigma_phi = 0.01
sigma_g = 0.05
sigma_p = 0.05
# 圆周距离函数
def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis = bm.where(dis>bm.pi, dis-2*bm.pi, dis)
    dis = bm.where(dis<-bm.pi, dis+2*bm.pi, dis)
    return dis
# feature space
x = np.linspace(0, Coding_range, num_p, endpoint=False)
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
num_module = len(Lambda)
z_truth = 62
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi 
psi = phi_truth + sigma_phi * np.random.randn(num_module)
psi = phi_truth 
Ig = np.zeros((num_module, num_g))
for i in range(num_module):
    dis_theta = circ_dis(theta, psi[i])
    Ig[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2)) + sigma_g * np.random.randn(num_g)
x = np.linspace(0,Coding_range,num_p,endpoint=False)
dis_x = x-z_truth
Ip = np.exp(-dis_x**2 / (4 * a_p**2)) + sigma_p * np.random.randn(num_p)


def plot_data(Ip,Ig):
    # 设置美化风格
    plt.style.use('seaborn-darkgrid')
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['axes.titlesize'] = 'x-large'

    # 前面的数据生成部分代码保持不变

    # 设置图形和子图网格
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3)

    # 第一行，一个子图占据所有三列
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(x, Ip.flatten(), label='Intensity profile')
    ax1.set_title('Place cells')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Activation')
    ax1.legend()

    # 第二行，三个子图分别占据一列
    colors = plt.cm.viridis(np.linspace(0, 1, num_module))  # 使用Colormap
    for i in range(num_module):
        ax = plt.subplot(gs[1, i])
        ax.plot(Ig[i, :], color=colors[i])
        ax.set_title(f'Grid cells, Spacing={Lambda[i]}')
        ax.set_xlabel('Spatial Phase')
        ax.set_ylabel('Activation')
### 实例化网络模型
### Place cells
P_CANN = Place_net(z_min=0, z_max=Coding_range, num=num_p, a_p=a_p, k=k_p, tau=tau_p, J0=J_p)
### Grid cells
G_CANNs = bm.NodeList()
for i in range(num_module):
    G_CANNs.append(Grid_net(z_min=0, z_max=Coding_range, num = num_g, num_hpc=num_p, L = Lambda[i], a_g = a_g[i], k_mec=k_g[i], tau=tau_g[i], J0=J_g, W0=J_pg))
### Coupled Network
Coupled_model = Coupled_Net(HPC_model=P_CANN, MEC_model_list=G_CANNs, num_module=num_module)

# run net functions

def initial_net(alpha_p, alpha_g, Ip, Ig): 
    Coupled_model.initial(alpha_p, alpha_g, Ip, Ig)
    u_HPC = Coupled_model.HPC_model.u
    u_grid = bm.zeros([num_module,num_g])
    for i in range(num_module):
        u_grid[i] = Coupled_model.MEC_model_list[i].u
    I_mec = Coupled_model.I_mec
    return u_HPC, u_grid, I_mec

def run_net(i, alpha_p, alpha_g, Ip, Ig): 
    Coupled_model.step_run(i, alpha_p, alpha_g, Ip, Ig)
    u_HPC = Coupled_model.HPC_model.u
    u_grid = bm.zeros([num_module,num_g])
    for i in range(num_module):
        u_grid[i] = Coupled_model.MEC_model_list[i].u
    I_mec = Coupled_model.I_mec
    phi_decode = Coupled_model.phase
    z_decode = Coupled_model.HPC_model.center
    energy =  Coupled_model.energy
    return u_HPC, u_grid, I_mec, z_decode, phi_decode, energy

# Initialize the network
T = 8000
indices = np.arange(T)
z0 = z_truth-1
phi_0 = np.mod(z0 / Lambda, 1) * 2 * np.pi
fg = np.zeros((num_module, num_g))
for i in range(num_module):
    dis_theta = circ_dis(theta, phi_0[i])
    fg[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2))
x = np.linspace(0,Coding_range,num_p,endpoint=False)
dis_x = x-z0
fp = np.exp(-dis_x**2 / (4 * a_p**2))
I_place = 1*np.repeat(fp[np.newaxis, :], T, axis=0)
I_grid = 1*np.repeat(fg[np.newaxis, :, :], T, axis=0)
alpha_p = np.zeros(T,) + 1
alpha_g = np.zeros(T,) + 1
I_place[int(T/2):,:] = 0
I_grid[int(T/2):,:,:] = 0
u_HPC_init, u_grid_init, I_mec_init = bm.for_loop(initial_net, (alpha_p, alpha_g, I_place, I_grid), progress_bar=True)


# input_strength
A_p = 0.05
A_g = 0.05

### 
T = 5000
indices = np.arange(T)
alpha_p = np.zeros(T,) + A_p
alpha_g = np.zeros(T,) + A_g


I_place = np.repeat(Ip[np.newaxis, :], T, axis=0)
I_grid = np.repeat(Ig[np.newaxis, :, :], T, axis=0)
u_HPC, u_grid, I_mec, z_decode, phi_decode, energy = bm.for_loop(run_net, (indices, alpha_p, alpha_g, I_place, I_grid), progress_bar=True)

