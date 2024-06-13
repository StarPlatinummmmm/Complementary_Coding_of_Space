
import math
import torch
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

## choose the number of grid modules
M = 3 # number of grid modules
## choose the strength of noise
sigma_ratio = 0.005

## common probability model simulation params
L = 200
dt = 0.1 # walking time step
v = 10.0 # walking speed
eta = 3*1e-6 # learning rate for GOP, 7*1e-6

## LSC params
n_p = 1600
rho_p = n_p / L # 1600/200 = 8
a_p = 0.3
J_p = 20. # LSC recurrent connection strength
k_p = 20. # divisive normalization
tau_p = 1. # time constant
alpha_p = 0.05 # sensory input strength

A_p = 1./(4*np.sqrt(math.pi)*a_p*rho_p*k_p)*(rho_p*J_p+np.sqrt((rho_p*J_p)**2-8*np.sqrt(2*math.pi)*a_p*rho_p*k_p))

R_p = A_p**2/(1+k_p*rho_p*a_p*np.sqrt(2*math.pi)*A_p**2)
R_p_np = R_p 
R_p = torch.tensor(R_p).to(device)
# sigma_p
sigma_p = np.sqrt(np.sqrt(math.pi)*A_p**3*rho_p*tau_p/(a_p*alpha_p)) * sigma_ratio
sigma_p = torch.tensor(sigma_p).to(device)

params_prob = {
        "L": L,
        "dt": dt,
        "v": v,
        "eta": eta,
        "sigma_ratio": sigma_ratio
    }

params_LSC = {
        "L": L,
        "n_p": n_p,
        "rho_p": rho_p,
        "a_p": a_p,
        "A_p": A_p,
        "R_p": R_p,
        "sigma_p": sigma_p,
        "J_p": J_p,
        "k_p": k_p,
        "tau_p": tau_p,
        "alpha_p": alpha_p,
    }

## common PSC params
Lphase = 2*math.pi
n_g = 50
rho_g = n_g / Lphase
J_g = J_p # PSC recurrent connection strength
alpha_g = 0.05 # path integration input strength

if M == 1:
    # grid spacing
    lambda_gs_np = np.array([210.])
    # reciprocal connection strength between LSC and PSC
    J_pg = J_p / 40 * np.ones([M]) 
elif M == 2:
    lambda_gs_np = np.array([10., 21.])
    J_pg = J_p / 40 * np.ones([M])
elif M == 3:
    lambda_gs_np = np.array([3., 7., 10.])
    J_pg = J_p / 40 * np.ones([M])
elif M == 4:
    lambda_gs_np = np.array([2., 3., 5., 7.])
    J_pg = J_p / 40 * np.ones([M])

# grid spacing
lambda_gs = torch.tensor(lambda_gs_np).to(device)
a_gs_np = params_LSC['a_p'] /lambda_gs_np * Lphase
a_gs = params_LSC['a_p'] /lambda_gs * Lphase
# speed in phase space
v_lambda_gs = params_prob['v'] / lambda_gs * Lphase

# divisive normalization
k_gs = lambda_gs_np / Lphase * k_p
# time constants
tau_gs = Lphase/lambda_gs_np * tau_p
# A_g,R_g
A_g = 1./(4*np.sqrt(np.pi)*a_gs_np*rho_g*k_gs)*(rho_g*J_g+np.sqrt((rho_g*J_g)**2-8*np.sqrt(2*math.pi)*a_gs_np*rho_g*k_gs))
R_g = A_g**2/(1+k_gs*rho_g*a_gs_np*np.sqrt(2*math.pi)*A_g**2)
R_g_np = R_g
R_g = torch.tensor(R_g).to(device)
# sigma_g
sigma_g = np.sqrt(np.sqrt(math.pi)*A_g**3*rho_g*tau_gs/(a_gs_np*alpha_g)) * sigma_ratio
sigma_g = torch.tensor(sigma_g).to(device)
# sigma_phi
sigma_phi = 1/((lambda_gs_np/Lphase) * np.sqrt(J_pg*rho_g*R_g_np/(4*A_p*tau_p))) * sigma_ratio
sigma_phi = torch.tensor(sigma_phi).to(device)

params_PSC = {
        "Lphase": Lphase,
        "n_g": n_g,
        "rho_g": rho_g,
        "M": M,
        "sigma_g": sigma_g,
        "lambda_gs_np": lambda_gs_np,
        "lambda_gs": lambda_gs,
        "a_gs_np":a_gs_np,
        "a_gs": a_gs,
        "A_g": A_g,
        "R_g": R_g,
        "v_lambda_gs": v_lambda_gs,
        "sigma_phi": sigma_phi,
        "J_g": J_g,
        "J_pg": J_pg,
        "k_gs": k_gs,
        "tau_gs": tau_gs,
        "alpha_g": alpha_g,
    }



if __name__ == '__main__':
    print('params_prob',params_prob)
    print('params_LSC',params_LSC)
    print('params_PSC',params_PSC)
    print('A_p',A_p)
    print('R_p',R_p)
    print('A_g',A_g)
    print('R_g',R_g)
    print('R_g_np',R_g_np)
    print('Done')