import numpy as np
import matplotlib.pyplot as plt
from Network import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from Functions import *

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

# 生成I_p, I_g
sigma_phi_generate = 0.5
sigma_g_generate = 0.1
sigma_p_generate = 0.1
# input_strength
alpha_p = 0.0
alpha_g = 1.
# exp number 
exp_number = 500
z_MAP = np.zeros(exp_number,)
z_GOP = np.zeros(exp_number,)
z_NET = np.zeros(exp_number,)
for ei in range(exp_number):

    Ip, Ig = generate_model(z_truth=30, sigma_phi_generate=sigma_phi_generate, 
                            sigma_g_generate=sigma_g_generate, sigma_p_generate=sigma_p_generate,
                            alpha_p=alpha_p, alpha_g=alpha_g, Lambda=Lambda)


    Ag = 1./(4*np.sqrt(np.pi)*a_g*rho_g*k_g)*(rho_g*J_g+np.sqrt((rho_g*J_g)**2-8*np.sqrt(2*np.pi)*a_g*rho_g*k_g))
    Ap = 1./(4*np.sqrt(np.pi)*a_p*rho_p*k_p)*(rho_p*J_p+np.sqrt((rho_p*J_p)**2-8*np.sqrt(2*np.pi)*a_p*rho_p*k_p))
    Rg = Ag**2/(1+k_g*rho_g*a_g*np.sqrt(2*np.pi)*Ag**2)

    sigma_g = np.sqrt(np.sqrt(np.pi)*Ag**3*rho_g*tau_g/(a_g*alpha_g + 1e-8)) 
    sigma_phi = 1/((Lambda/2/np.pi) * np.sqrt(J_pg*rho_g*Rg/(4*Ap*tau_p))) 
    sigma_p = np.sqrt(np.sqrt(np.pi)*Ap**3*rho_p*tau_p/(a_p*alpha_p + 1e-8)) 

    candidate_num = 1000
    Energy = Get_energy(sigma_g=sigma_g, sigma_phi=sigma_phi, sigma_p=sigma_p, Ig=Ig, Ip=Ip, candidate_num=candidate_num, Lambda=Lambda, a_g=a_g, a_p=a_p)

    z_MAP[ei] = MAP_decoding(sigma_g=sigma_g, sigma_phi=sigma_phi, sigma_p=sigma_p, Ig=Ig, Ip=Ip, candidate_num=candidate_num, Lambda=Lambda, a_g=a_g, a_p=a_p)

    z_init = 30
    z_GOP[ei] = GOP_decoding(z_init=z_init, sigma_g=sigma_g, sigma_phi=sigma_phi, sigma_p=sigma_p, Ig=Ig, Ip=Ip, candidate_num=candidate_num, Lambda=Lambda, a_g=a_g, a_p=a_p)
    
    u_HPC, u_grid, z_decode, phi_decode, energy = Net_decoding(alpha_p, alpha_g, Ip, Ig, z_init=z_init, 
                                                            Lambda=Lambda, a_p=a_p, a_g = a_g, 
                    num_p = num_p, k_p = k_p, tau_p = tau_p,
                    J_p = J_p, num_g = num_g, k_g=k_g, tau_g=tau_g, J_g=J_g, J_pg=J_pg)
    
    z_NET[ei] = z_decode[-1]
    print('progress:', ei/exp_number)
# 保存 z_MAP, z_GOP, z_NET 矩阵
np.save('z_MAP.npy', z_MAP)
np.save('z_GOP.npy', z_GOP)
np.save('z_NET.npy', z_NET)

# 绘制并美化图形
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].hist(z_MAP, bins=30, color='blue', edgecolor='black')
ax[0].set_title('z_MAP Distribution', fontsize=14)
ax[0].set_xlabel('z_MAP', fontsize=12)
ax[0].set_ylabel('Frequency', fontsize=12)

ax[1].hist(z_GOP, bins=30, color='green', edgecolor='black')
ax[1].set_title('z_GOP Distribution', fontsize=14)
ax[1].set_xlabel('z_GOP', fontsize=12)
ax[1].set_ylabel('Frequency', fontsize=12)

ax[2].hist(z_NET, bins=30, color='red', edgecolor='black')
ax[2].set_title('z_NET Distribution', fontsize=14)
ax[2].set_xlabel('z_NET', fontsize=12)
ax[2].set_ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.savefig('distributions.png')
plt.show()