import numpy as np
import matplotlib.pyplot as plt
from Network import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
import time
from multiprocessing import Pool
# 圆周距离函数
def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis = bm.where(dis>bm.pi, dis-2*bm.pi, dis)
    dis = bm.where(dis<-bm.pi, dis+2*bm.pi, dis)
    # dis[dis > np.pi] -= 2 * np.pi
    # dis[dis < -np.pi] += 2 * np.pi
    return dis
# 默认参数
# grid spacing
lambda_1 = 5
lambda_2 = 6
lambda_3 = 8
Lambda = np.array([lambda_1, lambda_2, lambda_3])
L = lambda_1 * lambda_2 * lambda_3
# cell number
num_p = int(1280)*2
rho_p = num_p/L
rho_g = rho_p
num_g = int(rho_g*2*np.pi) # 为了让两个网络的rho相等
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

# sigma
# Ag = bm.ones([M])
# Ap = 1.
Ag = 1./(4*np.sqrt(np.pi)*a_g*rho_g*k_g)*(rho_g*J_g+np.sqrt((rho_g*J_g)**2-8*np.sqrt(2*np.pi)*a_g*rho_g*k_g))
Ap = 1./(4*np.sqrt(np.pi)*a_p*rho_p*k_p)*(rho_p*J_p+np.sqrt((rho_p*J_p)**2-8*np.sqrt(2*np.pi)*a_p*rho_p*k_p))
Rp = Ap**2/(1+k_p*rho_p*a_p*np.sqrt(2*np.pi)*Ap**2)
Rg = Ag**2/(1+k_g*rho_g*a_g*np.sqrt(2*np.pi)*Ag**2)


sigma_g = np.sqrt(np.sqrt(np.pi)*Ag**3*rho_g*tau_g/(a_g*alpha_g)) 
sigma_phi = 1/((Lambda/2/np.pi) * np.sqrt(J_pg*rho_g*Rg/(4*Ap*tau_p))) 
sigma_p = np.sqrt(np.sqrt(np.pi)*Ap**3*rho_p*tau_p/(a_p*alpha_p)) 
ratio = 0.005
print('sigma_g:', sigma_g[0])
print('sigma_phi:', sigma_phi)

z_truth = 62
z_m = z_truth-0.1
z_e = z_truth+0.1

def Net_decoding(Ip, Ig, alpha_p = 0.05, alpha_g = 0.05):
    ### Place cells
    P_CANN = Place_net(z_min=0, z_max=L, num=num_p, a_p=a_p, k=k_p, tau=tau_p, J0=J_p)
    ### Grid cells
    G_CANNs = bm.NodeList()
    for i in range(M):
        G_CANNs.append(Grid_net(z_min=0, z_max=L, num = num_g, num_hpc=num_p, L = Lambda[i], a_g = a_g[i], k_mec=k_g[i], tau=tau_g[i], J0=J_g, W0=J_pg))
    ### Coupled Network
    Coupled_model = Coupled_Net(HPC_model=P_CANN, MEC_model_list=G_CANNs, num_module=M)

    def initial_net(Ip, Ig): 
        Coupled_model.initial(Ip, Ig)

    def run_net(i, Ip, Ig): 
        Coupled_model.step_run(i, Ip, Ig)
        phi_decode = Coupled_model.phase
        z_decode = Coupled_model.HPC_model.center
        return z_decode, phi_decode

    T_init = 1000
    z0 = (z_e+z_m)/2
    phi_0 = np.mod(z0 / Lambda, 1) * 2 * np.pi
    fg = np.zeros((M, num_g))
    for i in range(M):
        dis_theta = circ_dis(theta, phi_0[i])
        fg[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2))
    x = np.linspace(0,L,num_p,endpoint=False)
    dis_x = x-z0
    fp = np.exp(-dis_x**2 / (4 * a_p**2))
    I_place = 1*np.repeat(fp[np.newaxis, :], T_init, axis=0)
    I_grid = 1*np.repeat(fg[np.newaxis, :, :], T_init, axis=0)
    I_place[int(T_init/2):,:] = 0
    I_grid[int(T_init/2):,:,:] = 0

    bm.for_loop(initial_net, (I_place, I_grid), progress_bar=False)
    T = 4000
    indices = np.arange(T)
    I_place = alpha_p*np.repeat(Ip[np.newaxis, :], T, axis=0)
    I_grid = alpha_g*np.repeat(Ig[np.newaxis, :, :], T, axis=0)
    z_record, phi_record = bm.for_loop(run_net, (indices, I_place, I_grid), progress_bar=False)
    return z_record[-1], phi_record[-1]

## Generative model
trial_num = 10
z_decode_env = np.zeros(trial_num)
z_decode_mot = np.zeros(trial_num)
z_decode_both = np.zeros(trial_num)

start_time = time.time()
z_truth = 62
z_m = z_truth-0.1
phi_truth = np.mod(z_m / Lambda, 1) * 2 * np.pi 
z_e = z_truth+0.1
for i in range(trial_num):
    psi = phi_truth + sigma_phi * np.random.randn(M) * ratio
    psi = phi_truth 
    Ig = np.zeros((M, num_g))
    for j in range(M):
        dis_theta = circ_dis(theta, psi[j])
        Ig[j, :] = np.exp(-dis_theta**2 / (4 * a_g[j] ** 2)) + sigma_g[j] * ratio * np.random.randn(num_g)
    x = np.linspace(0,L,num_p,endpoint=False)
    dis_x = x-z_e
    Ip = np.exp(-dis_x**2 / (4 * a_p**2)) + sigma_p * np.random.randn(num_p) * ratio

    z_decode_e, _ = Net_decoding(Ip, Ig, alpha_g=0)
    z_decode_env[i] = z_decode_e
    z_decode_m, _ = Net_decoding(Ip, Ig, alpha_p=0)
    z_decode_mot[i] = z_decode_m
    z_decode_b, _ = Net_decoding(Ip, Ig)
    z_decode_both[i] = z_decode_b
    print(f'Progress: {i+1}/{trial_num}')


# 保存矩阵
np.savez('decoding_results.npz', z_decode_env=z_decode_env, z_decode_mot=z_decode_mot, z_decode_both=z_decode_both)

print(z_decode_env)
print(z_decode_mot)
print(z_decode_both)

end_time = time.time()
print(f'计算时间: {end_time - start_time:.2f} 秒')
