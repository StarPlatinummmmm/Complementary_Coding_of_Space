
import math
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M = 2
if M == 1:
    L = 100

elif M == 2:
    # 2 modules
    ## probability model simulation params
    L = 70.
    dt = 1.0
    v = 2.
    eta = 2*1e-4 # learning rate

    params_prob = {
        "L": L,
        "dt": dt,
        "v": v,
        "eta": eta,
    }

    ## LSC params
    n_p = 256
    rho_p = n_p / L
    a_p = 0.8
    R_p = 1.0
    sigma_p = 0.1
    params_LSC = {
        "L": L,
        "n_p": n_p,
        "rho_p": rho_p,
        "a_p": a_p,
        "R_p": R_p,
        "sigma_p": sigma_p,
    }

    # PSC params
    Lphase = 2*math.pi
    n_g = 50
    M = 2
    rho_g = n_g / Lphase
    R_g = 1.0
    sigma_g = 0.1
    # grid spacing
    lambda_gs_np = np.array([7., 11.])
    # lambda_gs = torch.tensor([7., 11.]).to(device)
    lambda_gs = torch.tensor(lambda_gs_np).to(device)
    a_gs = params_LSC['a_p'] /lambda_gs * Lphase

    # speed in phase space
    v_lambda_gs = params_prob['v'] / lambda_gs * Lphase
    # transition uncertainty
    sigma_phi_np = np.array([0.4, 0.4])
    # sigma_phi = torch.tensor([0.4, 0.4]).to(device)
    sigma_phi = torch.tensor(sigma_phi_np).to(device)

    params_PSC = {
        "Lphase": Lphase,
        "n_g": n_g,
        "rho_g": rho_g,
        "M": M,
        "sigma_g": sigma_g,
        "lambda_gs": lambda_gs,
        "a_gs": a_gs,
        "R_g": R_g,
        "v_lambda_gs": v_lambda_gs,
        "sigma_phi": sigma_phi,
    }

elif M == 3:
    # 3 modules
    ## probability model simulation params
    L = 1200.
    dt = 1.0
    v = 2.
    eta = 1e-5 # learning rate

    params_prob = {
        "L": L,
        "dt": dt,
        "v": v,
        "eta": eta,
    }

    ## LSC params
    n_p = 4800
    rho_p = n_p / L
    a_p = 0.4
    R_p = 1.0
    sigma_p = 0.1
    params_LSC = {
        "L": L,
        "n_p": n_p,
        "rho_p": rho_p,
        "a_p": a_p,
        "R_p": R_p,
        "sigma_p": sigma_p,
    }

    # PSC params
    Lphase = 2*math.pi
    n_g = 50
    M = 3
    rho_g = n_g / Lphase
    R_g = 1.0
    sigma_g = 0.1
    # grid spacing
    lambda_gs_np = np.array([23., 11., 5.])
    # lambda_gs = torch.tensor([23., 11., 5.]).to(device)
    lambda_gs = torch.tensor(lambda_gs_np).to(device)
    a_gs = params_LSC['a_p'] /lambda_gs * Lphase

    # speed in phase space
    v_lambda_gs = params_prob['v'] / lambda_gs * Lphase
    # transition uncertainty
    sigma_phi_np = np.array([0.02, 0.02, 0.02])
    # sigma_phi = torch.tensor([0.02, 0.02, 0.02]).to(device)
    sigma_phi = torch.tensor(sigma_phi_np).to(device)

    params_PSC = {
        "Lphase": Lphase,
        "n_g": n_g,
        "rho_g": rho_g,
        "M": M,
        "sigma_g": sigma_g,
        "lambda_gs": lambda_gs,
        "a_gs": a_gs,
        "R_g": R_g,
        "v_lambda_gs": v_lambda_gs,
        "sigma_phi": sigma_phi,
    }

if __name__ == '__main__':
    print('params')
    print()