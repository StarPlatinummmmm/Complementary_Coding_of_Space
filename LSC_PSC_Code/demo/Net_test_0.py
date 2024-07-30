import numpy as np
import matplotlib.pyplot as plt
from Network import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm
from matplotlib.animation import FuncAnimation

# 圆周距离函数
def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis[dis > np.pi] -= 2 * np.pi
    dis[dis < -np.pi] += 2 * np.pi
    return dis

# 默认参数
# grid spacing
lambda_1 = 4
lambda_2 = 5
lambda_3 = 6
Lambda = np.array([lambda_1, lambda_2, lambda_3])
L = lambda_1 * lambda_2 * lambda_3
# cell number
num_p = 1280
rho_p = num_p/L
rho_g = rho_p
num_g = int(rho_g*2*np.pi) # 为了让两个网络的rho相等
M = len(Lambda)
# feature space
x = np.linspace(0, L, num_p, endpoint=False)
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
# connection range
a_p = 0.4
a_g = a_p/Lambda*2*np.pi
# connection strength
J_p = 20
J_g = J_p
J_pg = J_p/40
# divisive normalization
k_p = 10.
k_g = Lambda/2/np.pi * k_p
# time constants
tau_p = 1.
# tau_g = Lambda/2/np.pi * tau_p
tau_g = 2*np.pi/Lambda * tau_p
# input_strength
alpha_p = 0.
alpha_g = 0.1

# sigma
# Ag = 1.
# Ap = 1.
# Rg = 1.
# Rp = 1.
# sigma_g = np.sqrt(np.sqrt(2*np.pi)*(Ag**2)*rho_g*tau_g/(4*a_g*alpha_g))
# sigma_phi = np.sqrt(np.sqrt(2)*(Ag)*tau_g/(4*J_pg*rho_p*Rp))
sigma_g = 0.1
sigma_phi = 0.1
sigma_p = 0.1
# print(sigma_g, sigma_phi)

z_truth = 62
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi
psi = phi_truth + sigma_phi * np.random.randn(M)
Ig = np.zeros((M, num_g))
for i in range(M):
    dis_theta = circ_dis(theta, psi[i])
    Ig[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2)) + sigma_g * np.random.randn(num_g)
x = np.linspace(0,L,num_p,endpoint=False)
dis_x = x-z_truth
Ip = np.exp(-dis_x**2 / (4 * a_p**2)) + sigma_p * np.random.randn(num_p)

### Place cells
P_CANN = Place_net(z_min=0, z_max=L, num=num_p, a_p=a_p, k=k_p, tau=tau_p, J0=J_p)
### Grid cells
G_CANNs = bm.NodeList()
for i in range(M):
    G_CANNs.append(Grid_net(z_min=0, z_max=L, num = num_g, num_hpc=num_p, L = Lambda[i], a_g = a_g[i], k_mec=k_g[i], tau=tau_g[i], J0=J_g, W0=J_pg))
### Coupled Network
Coupled_model = Coupled_Net(HPC_model=P_CANN, MEC_model_list=G_CANNs, num_module=M)


def Net_decoding(Ip, Ig,Coupled_model):
    def initial_net(Ip, Ig): 
        Coupled_model.initial(Ip, Ig)
        u_HPC = Coupled_model.HPC_model.u
        u_grid = Coupled_model.MEC_model_list[0].u
        I_mec = Coupled_model.I_mec
        return u_HPC, u_grid, I_mec

    def run_net(i, Ip, Ig): 
        Coupled_model.step_run(i, Ip, Ig)
        u_HPC = Coupled_model.HPC_model.u
        u_grid = Coupled_model.MEC_model_list[0].u
        I_mec = Coupled_model.I_mec
        phi_decode = Coupled_model.phase
        z_decode = Coupled_model.HPC_model.center
        return u_HPC, u_grid, I_mec, z_decode, phi_decode

    T = 1000
    indices = np.arange(T)
    z0 = 60.1
    phi_0 = np.mod(z0 / Lambda, 1) * 2 * np.pi
    fg = np.zeros((M, num_g))
    for i in range(M):
        dis_theta = circ_dis(theta, phi_0[i])
        fg[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2))
    x = np.linspace(0,L,num_p,endpoint=False)
    dis_x = x-z0
    fp = np.exp(-dis_x**2 / (4 * a_p**2))
    I_place = 1*np.repeat(fp[np.newaxis, :], T, axis=0)
    I_grid = 1*np.repeat(fg[np.newaxis, :, :], T, axis=0)
    I_place[int(T/2):,:] = 0
    I_grid[int(T/2):,:,:] = 0
    u_HPC, u_grid, I_mec = bm.for_loop(initial_net, (I_place, I_grid), progress_bar=True)

    ### 
    T = 5000
    indices = np.arange(T)
    

    I_place = alpha_p*np.repeat(Ip[np.newaxis, :], T, axis=0)
    I_grid = alpha_g*np.repeat(Ig[np.newaxis, :, :], T, axis=0)
    u_HPC, u_grid, I_mec, z_record, phi_record = bm.for_loop(run_net, (indices, I_place, I_grid), progress_bar=True)

    fontsize = 20
    linewidth = 2.5

    plt.figure()
    for i in range(M):
        plt.plot(theta, fg[i], linewidth=linewidth)
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
    plt.ylim([59, 63])
    plt.tight_layout()
    plt.savefig('figure2.png')

    plt.figure()
    for i in range(M):
        # plt.plot(phi_record[:, i]+np.pi, linewidth=linewidth)
        plt.plot(phi_record[:, i], linewidth=linewidth)
        plt.plot([0, T], [phi_truth[i], phi_truth[i]], '--', linewidth=linewidth)
    # plt.plot(phi_record[:, 1]+np.pi, linewidth=linewidth)
    # plt.plot([0, T], [phi_truth[1], phi_truth[1]], '--', linewidth=linewidth)
    plt.xlabel('T', fontsize=12)
    plt.ylabel('Phase φ', fontsize=fontsize)
    plt.gca().tick_params(width=linewidth)
    plt.gca().tick_params(labelsize=fontsize)
    plt.savefig('figure3.png')
    # plt.show()

    return z_record, phi_record, u_HPC, u_grid, I_mec



z_record, phi_record, u_HPC, u_grid, I_mec = Net_decoding(Ip, Ig, Coupled_model)



animate = 1
if animate == 1:
    x_hpc = P_CANN.x
    x_grid = G_CANNs[0].x
    n_step = 50
    # 创建动画
    # 动画化网络活动
    data1 = u_HPC[::n_step, :]
    data2 = I_mec[::n_step, :]
    data3 = u_grid[::n_step, :]
    # data3 = I_mec[::n_step, :]
    # data4 = r_sen[:, center_sensory_index]
    N = data1.shape[1]
    T = data1.shape[0]
    # 创建画布和轴
    fig, ax_ani = plt.subplots(1,2)
    ax_ani[0].set_xlim(0, L)
    ax_ani[0].set_ylim(-0.01, 2.5)
    # ax_ani[1].set_xlim(-np.pi, np.pi)
    ax_ani[1].set_xlim(0, 2*np.pi)
    ax_ani[1].set_ylim(-0.01, 5)
    # 创建初始空白线条
    line1, = ax_ani[0].plot([], [])
    line2, = ax_ani[0].plot([], [])
    line3, = ax_ani[1].plot([], [])
    ax_ani[1].plot(x_grid, Ig[0])
    ax_ani[0].legend(['fr-HPC', 'I-mec'])
    ax_ani[0].set_title('Population activities')

    # 更新线条的函数
    def update(frame):
        y1 = data1[frame].flatten()
        # y1 = y1 / np.max(y1)
        y2 = data2[frame].flatten()
        # y2 = y2 / np.max(y2)
        y3 = data3[frame].flatten()
        # y3 = y3 / np.max(y3)
        line1.set_data(x_hpc, y1)
        line2.set_data(x_hpc, y2)
        line3.set_data(x_grid, y3)
        # line3.set_data(Center_place, y3)
        # line4.set_data(Center_sensory, y4)
        return line1, line2, line3 

    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=True)
plt.show()
