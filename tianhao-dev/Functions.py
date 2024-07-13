import numpy as np
import matplotlib.pyplot as plt
from Network import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

# Parameters
# grid spacing
lambda_1 = 3
lambda_2 = 4
lambda_3 = 5
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


# 圆周距离函数
def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis = bm.where(dis>bm.pi, dis-2*bm.pi, dis)
    dis = bm.where(dis<-bm.pi, dis+2*bm.pi, dis)
    return dis

def generate_model(z_truth, sigma_phi_generate, sigma_g_generate, sigma_p_generate, alpha_p, alpha_g, Lambda=Lambda, a_g=a_g, a_p=a_p, num_g = num_g, num_p= num_p):
    Coding_range = np.prod(Lambda)
    num_module = len(Lambda)
    theta = np.linspace(-np.pi,np.pi,num_g, endpoint=False)
    x = np.linspace(0, Coding_range, num_p, endpoint=False)

    phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi 
    psi = phi_truth + sigma_phi_generate * np.random.randn(num_module)
    # psi = phi_truth 
    Ig = np.zeros((num_module, num_g))
    for i in range(num_module):
        dis_theta = circ_dis(theta, psi[i])
        Ig[i, :] = alpha_g*(np.exp(-dis_theta**2 / (4 * a_g[i] ** 2)) + sigma_g_generate * np.random.randn(num_g))

    dis_x = x-z_truth
    Ip = alpha_p*(np.exp(-dis_x**2 / (4 * a_p**2)) + sigma_p_generate * np.random.randn(num_p))
    return Ip, Ig

# Calculate A, R based on theoretical formula
def Get_energy(sigma_g, sigma_phi, sigma_p, Ig, Ip, candidate_num=1000, Lambda=Lambda, a_g=a_g, a_p=a_p):
    M = len(Lambda)
    L = np.prod(Lambda)
    num_p = len(Ip)
    num_g = Ig.shape[1]

    z_candidate = np.linspace(0, L , candidate_num)
    phi_candidate = np.mod(z_candidate.reshape(candidate_num,1) / Lambda.reshape(1,M), 1) * 2 * np.pi
    Energy = np.zeros(candidate_num,)
    # 后验概率计算函数

    def circ_dis_L(x1, x2, L):
        dis = x1 - x2
        dis = bm.where(dis>L/2, dis-L, dis)
        dis = bm.where(dis<-L/2, dis+L, dis)
        return dis

    def calculate_posterior(z, phi, Ig, Ip):
        x = np.linspace(0,L,num_p,endpoint=False)
        theta = np.linspace(0,2*np.pi,num_g,endpoint=False)
        psi_z = np.mod(z / Lambda, 1) * 2 * np.pi
        log_prior = 0
        log_likelihood_grid = 0
        for i in range(M):
            dis_1 = circ_dis(theta, phi[i])
            fg = np.exp(-dis_1**2 / (4 * a_g[i]**2))

            log_likelihood_grid -= np.sum((Ig[i, :] - fg)**2) / sigma_g[i]**2
            dis_2 = circ_dis(phi[i], psi_z[i])
            log_prior -= 1 / (sigma_phi[i]**2) * np.exp(-dis_2**2/8/a_g[i]**2) * dis_2**2
        dis_x = circ_dis_L(x, z, L)
        fp = np.exp(-dis_x**2 / (4 * a_p ** 2))
        log_likelihood_place = -np.sum((Ip - fp)**2) / sigma_p**2
        log_posterior = log_likelihood_grid + log_prior + log_likelihood_place
        # log_posterior = log_likelihood_place

        return log_posterior

    # 计算能量
    for i in range(candidate_num):
        Energy[i] = -calculate_posterior(z_candidate[i], phi_candidate[i], Ig, Ip)
    return Energy

def MAP_decoding(sigma_g, sigma_phi, sigma_p, Ig, Ip, candidate_num, Lambda=Lambda, a_g=a_g, a_p=a_p):
    Energy = Get_energy(sigma_g, sigma_phi, sigma_p, Ig, Ip, candidate_num, Lambda, a_g, a_p)
    L = np.prod(Lambda)
    z_candidate = np.linspace(0, L , candidate_num)
    global_minimum = np.argmin(Energy)
    z_decode = z_candidate[global_minimum]
    
    return z_decode

import numpy as np
from scipy.signal import find_peaks

def GOP_decoding(z_init, sigma_g, sigma_phi, sigma_p, Ig, Ip, candidate_num=1000, Lambda=None, a_g=None, a_p=None):

    # 计算能量
    Energy = Get_energy(sigma_g, sigma_phi, sigma_p, Ig, Ip, candidate_num, Lambda, a_g, a_p)
    L = np.prod(Lambda)
    z_candidate = np.linspace(0, L, candidate_num)
    
    # 找到局部最小值
    peaks, _ = find_peaks(-Energy)
    local_minima = peaks
    
    # 计算z_init处的梯度
    dz = z_candidate[1] - z_candidate[0]
    gradient = np.gradient(Energy, dz)
    init_gradient = np.interp(z_init, z_candidate, gradient)
    
    # 找到离 z_init 最近的局部最小值
    if init_gradient > 0:
        # 梯度为正，找右侧最近的局部最小值
        right_local_minima = local_minima[z_candidate[local_minima] > z_init]
        if len(right_local_minima) > 0:
            closest_local_minimum_idx = np.argmin(z_candidate[right_local_minima] - z_init)
            closest_local_minimum = right_local_minima[closest_local_minimum_idx]
        else:
            closest_local_minimum = local_minima[np.argmin(np.abs(z_candidate[local_minima] - z_init))]
    else:
        # 梯度为负，找左侧最近的局部最小值
        left_local_minima = local_minima[z_candidate[local_minima] < z_init]
        if len(left_local_minima) > 0:
            closest_local_minimum_idx = np.argmin(z_init - z_candidate[left_local_minima])
            closest_local_minimum = left_local_minima[closest_local_minimum_idx]
        else:
            closest_local_minimum = local_minima[np.argmin(np.abs(z_candidate[local_minima] - z_init))]
    
    z_decode = z_candidate[closest_local_minimum]
    
    return z_decode


def Net_decoding(alpha_p, alpha_g, Ip, Ig, z_init, Lambda=Lambda, a_p=a_p, a_g = a_g, 
                 num_p = num_p, k_p = k_p, tau_p = tau_p,
                 J_p = J_p, num_g = num_g, k_g=k_g, tau_g=tau_g, J_g=J_g, J_pg=J_pg):
    ### 实例化网络模型
    num_module = len(Lambda)
    Coding_range = np.prod(Lambda)
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

    def run_net(i, alpha_p, alpha_g, Ip, Ig): 
        Coupled_model.step_run(i, alpha_p, alpha_g, Ip, Ig)
        u_HPC = Coupled_model.HPC_model.u
        u_grid = bm.zeros([num_module,num_g])
        for i in range(num_module):
            u_grid[i] = Coupled_model.MEC_model_list[i].u
        phi_decode = Coupled_model.phase
        z_decode = Coupled_model.HPC_model.center
        energy =  Coupled_model.energy
        return u_HPC, u_grid, z_decode, phi_decode, energy
    
    # Initialize the network
    T = 8000
    indices = np.arange(T)
    z0 = z_init
    phi_0 = np.mod(z0 / Lambda, 1) * 2 * np.pi
    fg = np.zeros((num_module, num_g))
    theta = np.linspace(-np.pi, np.pi, num_g, endpoint=False)
    for i in range(num_module):
        dis_theta = circ_dis(theta, phi_0[i])
        fg[i, :] = np.exp(-dis_theta**2 / (4 * a_g[i] ** 2))
    x = np.linspace(0,Coding_range,num_p,endpoint=False)
    dis_x = x-z0
    fp = np.exp(-dis_x**2 / (4 * a_p**2))
    I_place = 1*np.repeat(fp[np.newaxis, :], T, axis=0)
    I_grid = 1*np.repeat(fg[np.newaxis, :, :], T, axis=0)
    A_p = np.zeros(T,) + 1
    A_g = np.zeros(T,) + 1
    I_place[int(T/2):,:] = 0
    I_grid[int(T/2):,:,:] = 0

    bm.for_loop(initial_net, (A_p, A_g, I_place, I_grid), progress_bar=False)


    ### 
    T = 5000
    indices = np.arange(T)
    alpha_p = np.zeros(T,) + alpha_p
    alpha_g = np.zeros(T,) + alpha_g


    I_place = np.repeat(Ip[np.newaxis, :], T, axis=0)
    I_grid = np.repeat(Ig[np.newaxis, :, :], T, axis=0)
    u_HPC, u_grid, z_decode, phi_decode, energy = bm.for_loop(run_net, (indices, alpha_p, alpha_g, I_place, I_grid), progress_bar=False)
    return u_HPC, u_grid, z_decode, phi_decode, energy





