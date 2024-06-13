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
lambda_1 = 3
lambda_2 = 7
lambda_3 = 10
Lambda = np.array([lambda_1, lambda_2, lambda_3])
L = 200
# cell number
num_p = 1600
rho_p = num_p/L
rho_g = rho_p
print('rho_p:',rho_p)
num_g = int(rho_g*2*np.pi) # 为了让两个网络的rho相等
print('num_g:',num_g)
M = len(Lambda)
# feature space
x = np.linspace(0, L, num_p, endpoint=False)
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
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
# input_strength
alpha_p = 0.05
alpha_g = 0.05

# Ap
Ap = 1./(4*np.sqrt(np.pi)*a_p*rho_p*k_p)*(rho_p*J_p+np.sqrt((rho_p*J_p)**2-8*np.sqrt(2*np.pi)*a_p*rho_p*k_p))
# Ag
Ag = 1./(4*np.sqrt(np.pi)*a_g*rho_g*k_g)*(rho_g*J_g+np.sqrt((rho_g*J_g)**2-8*np.sqrt(2*np.pi)*a_g*rho_g*k_g))

Rp = Ap**2/(1+k_p*rho_p*a_p*np.sqrt(2*np.pi)*Ap**2)

Rg = Ag**2/(1+k_g*rho_g*a_g*np.sqrt(2*np.pi)*Ag**2)

sigma_g = np.sqrt(np.sqrt(np.pi)*Ag**3*rho_g*tau_g/(a_g*alpha_g)) * 0.01
sigma_phi = 1/((Lambda/2/np.pi) * np.sqrt(J_pg*rho_g*Rg/(4*Ap*tau_p))) * 0.01

sigma_p = np.sqrt(np.sqrt(np.pi)*Ap**3*rho_p*tau_p/(a_p*alpha_p)) * 0.01

print('Ap:',Ap)
print('Ag:', Ag)
print('Rp:', Rp)
print('Rg:', Rg)
print('sigma_p', sigma_p)
print('sigma_g:', sigma_g)
print('sigma_phi:', sigma_phi)

# sigma_p = 0.1
# z_truth = 62
# phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi
# psi = phi_truth + sigma_phi * np.random.randn(M)
# Ig = np.zeros((M, num_g))
# for i in range(M):
#     dis_theta = circ_dis(theta, psi[i])
#     Ig[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2)) + sigma_g[i] * np.random.randn(num_g)
# x = np.linspace(0,L,num_p,endpoint=False)
# dis_x = x-z_truth
# Ip = np.exp(-dis_x**2 / (4 * a_p**2)) + sigma_p * np.random.randn(num_p)

# def Net_decoding(Ip, Ig):
#     ### Place cells
#     P_CANN = Place_net(z_min=0, z_max=L, num=num_p, a_p=a_p, k=k_p, tau=tau_p, J0=J_p)
#     ### Grid cells
#     G_CANNs = bm.NodeList()
#     for i in range(M):
#         G_CANNs.append(Grid_net(z_min=0, z_max=L, num = num_g, num_hpc=num_p, L = Lambda[i], a_g = a_g[i], k_mec=k_g[i], tau=tau_g[i], J0=J_g, W0=J_pg))
#     ### Coupled Network
#     Coupled_model = Coupled_Net(HPC_model=P_CANN, MEC_model_list=G_CANNs, num_module=M)


#     def initial_net(Ip, Ig): 
#         Coupled_model.initial(Ip, Ig)
#         u_HPC = Coupled_model.HPC_model.u
#         u_grid = Coupled_model.MEC_model_list[0].u
#         I_mec = Coupled_model.I_mec
#         return u_HPC, u_grid, I_mec

#     def run_net(i, Ip, Ig): 
#         Coupled_model.step_run(i, Ip, Ig)
#         u_HPC = Coupled_model.HPC_model.u
#         u_grid_1 = Coupled_model.MEC_model_list[0].u
#         u_grid_2 = Coupled_model.MEC_model_list[1].u
#         u_grid_3 = Coupled_model.MEC_model_list[2].u
#         I_mec = Coupled_model.I_mec
#         phi_decode = Coupled_model.phase
#         z_decode = Coupled_model.HPC_model.center
#         return u_HPC, u_grid_1, u_grid_2, u_grid_3, I_mec, z_decode, phi_decode



#     T = 5000
#     indices = np.arange(T)
#     z0 = 61
#     phi_0 = np.mod(z0 / Lambda, 1) * 2 * np.pi
#     fg = np.zeros((M, num_g))
#     for i in range(M):
#         dis_theta = circ_dis(theta, phi_0[i])
#         fg[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2))
#     x = np.linspace(0,L,num_p,endpoint=False)
#     dis_x = x-z0
#     fp = np.exp(-dis_x**2 / (4 * a_p**2))
#     I_place = 1*np.repeat(fp[np.newaxis, :], T, axis=0)
#     I_grid = 1*np.repeat(fg[np.newaxis, :, :], T, axis=0)
#     I_place[int(T/2):,:] = 0
#     I_grid[int(T/2):,:,:] = 0
#     u_HPC_init, u_grid_init, I_mec_init = bm.for_loop(initial_net, (I_place, I_grid), progress_bar=True)
    
    
#     ### 
#     T = 5000
#     indices = np.arange(T)
    

#     I_place = alpha_p*np.repeat(Ip[np.newaxis, :], T, axis=0)
#     I_grid = alpha_g*np.repeat(Ig[np.newaxis, :, :], T, axis=0)
#     u_HPC, u_grid_1, u_grid_2, u_grid_3, I_mec, z_record, phi_record = bm.for_loop(run_net, (indices, I_place, I_grid), progress_bar=True)
#     print('Theoretical bump height:', Ag[0])
#     print('Simulated bump height:', np.max(u_grid_1[-1]))

#     fontsize = 20
#     linewidth = 2.5

#     plt.figure()
#     for i in range(M):
#         plt.plot(theta, Ig[i], linewidth=linewidth)
#     plt.gca().tick_params(width=linewidth)
#     plt.gca().tick_params(labelsize=fontsize)
#     plt.savefig('Noisy_inputs.png')

#     plt.figure()
#     plt.plot(z_record, 'b', linewidth=linewidth)
#     plt.plot([0, T], [z_truth, z_truth], 'r--', linewidth=linewidth)
#     plt.xlabel('T', fontsize=12)
#     plt.ylabel('location z', fontsize=fontsize)
#     plt.gca().tick_params(width=linewidth)
#     plt.gca().tick_params(labelsize=fontsize)
#     plt.ylim([60.5, 63])
#     plt.tight_layout()
#     plt.savefig('Decoding_z.png')

#     plt.figure()
#     for i in range(M):
#         plt.plot(phi_record[:, i]+np.pi, linewidth=linewidth)
#         plt.plot([0, T], [phi_truth[i], phi_truth[i]], '--', linewidth=linewidth)
#         # plt.plot([0, T], [psi[i], psi[i]], '--', linewidth=linewidth)
#     # plt.plot(phi_record[:, 1]+np.pi, linewidth=linewidth)
#     # plt.plot([0, T], [phi_truth[1], phi_truth[1]], '--', linewidth=linewidth)
#     plt.xlabel('T', fontsize=12)
#     plt.ylabel('Phase φ', fontsize=fontsize)
#     plt.gca().tick_params(width=linewidth)
#     plt.gca().tick_params(labelsize=fontsize)
#     plt.savefig('Decoding_phi.png')
#     # plt.show()



#     animate = 1
#     if animate == 1:
#         x_hpc = P_CANN.x
#         x_grid = G_CANNs[0].x
#         n_step = 50
#         # 创建动画
#         # 动画化网络活动
#         data1 = u_HPC[::n_step, :]
#         data2 = 10*I_mec[::n_step, :]
#         data3 = u_grid_1[::n_step, :]
#         data4 = u_grid_2[::n_step, :]
#         data5 = u_grid_3[::n_step, :]
#         # data3 = I_mec[::n_step, :]
#         # data4 = r_sen[:, center_sensory_index]
#         N = data1.shape[1]
#         T = data1.shape[0]
#         # 创建画布和轴
#         y_lim = 1.8
#         fig, ax_ani = plt.subplots(2,2)
#         ax_ani[0][0].set_xlim(40, 80)
#         ax_ani[0][0].set_ylim(-1, y_lim)
#         ax_ani[0][1].set_xlim(-np.pi, np.pi)
#         ax_ani[0][1].set_ylim(-1, y_lim)
#         ax_ani[1][0].set_xlim(-np.pi, np.pi)
#         ax_ani[1][0].set_ylim(-1, y_lim)
#         ax_ani[1][1].set_xlim(-np.pi, np.pi)
#         ax_ani[1][1].set_ylim(-1, y_lim)
#         # 创建初始空白线条
#         line1, = ax_ani[0][0].plot([], [])
#         line2, = ax_ani[0][0].plot([], [])
#         line3, = ax_ani[0][1].plot([], [])
#         line4, = ax_ani[1][0].plot([], [])
#         line5, = ax_ani[1][1].plot([], [])
#         ax_ani[0][1].plot(x_grid, Ig[0])
#         ax_ani[1][0].plot(x_grid, Ig[1])
#         ax_ani[1][1].plot(x_grid, Ig[2])
#         ax_ani[0][0].legend(['u_p', 'W_gp r_g'])
#         ax_ani[0][1].legend(['u_g', 'W_gp r_p'])
#         ax_ani[1][0].legend(['u_g', 'W_gp r_p'])
#         ax_ani[1][1].legend(['u_g', 'W_gp r_p'])
#         # ax_ani[0][0].set_title('Population activities')

#         # 更新线条的函数
#         def update(frame):
#             y1 = data1[frame].flatten()
#             # y1 = y1 / np.max(y1)
#             y2 = data2[frame].flatten()
#             # y2 = y2 / np.max(y2)
#             y3 = data3[frame].flatten()
#             y4 = data4[frame].flatten()
#             y5 = data5[frame].flatten()
#             # y3 = y3 / np.max(y3)
#             line1.set_data(x_hpc, y1)
#             line2.set_data(x_hpc, y2)
#             line3.set_data(x_grid, y3)
#             line4.set_data(x_grid, y4)
#             line5.set_data(x_grid, y5)
#             # line3.set_data(Center_place, y3)
#             # line4.set_data(Center_sensory, y4)
#             return line1, line2, line3, line4, line5

#         ani = FuncAnimation(fig, update, frames=T, interval=20, blit=True)
#     plt.show()

#     return z_record[-1], phi_record[-1]

# z_decode, phi_decode = Net_decoding(Ip, Ig)