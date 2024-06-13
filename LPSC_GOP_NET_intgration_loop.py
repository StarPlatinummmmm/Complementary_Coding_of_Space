
import numpy as np
import math
import torch
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import argparse
import os,sys
import time

from utils import metric
from utils.logger import Logger
from utils.helperFunctions import circ_dis
from utils.params import params_prob, params_LSC, params_PSC
from utils.ProbModel import position2phase_one_module, position2phase_modules
from utils.Network import Place_net, Grid_net, Coupled_Net
from utils.generation import LSC, PSC

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def NET_decoder(Coupled_model, total_brainpy_iteration, z_t, phi_t, fp_t, fg_t, activation_p, activation_gs,alpha_p_infer,T_init = 800):
    def initial_net(Ip, Ig): 
        Coupled_model.initial(Ip, Ig)
        u_HPC = Coupled_model.HPC_model.u
        u_grid = Coupled_model.MEC_model_list[0].u
        I_mec = Coupled_model.I_mec
        return u_HPC, u_grid, I_mec
    
    def run_net(i, Ip, Ig, M = params_PSC['M']): 
        Coupled_model.step_run(i, Ip, Ig)
        u_HPC = Coupled_model.HPC_model.u
        u_grids = [Coupled_model.MEC_model_list[i].u for i in range(M)]
        # u_grid = Coupled_model.MEC_model_list[0].u
        I_mec = Coupled_model.I_mec
        phi_decode = Coupled_model.phase
        z_decode = Coupled_model.HPC_model.center
        return u_HPC, u_grids, I_mec, z_decode, phi_decode
    
    # initialize the brainpy network
    indices_init = np.arange(T_init)
    Ip_t = 1.* np.repeat(fp_t[np.newaxis, :], T_init, axis=0)
    Ig_t = 1.* np.repeat(fg_t[np.newaxis, :], T_init, axis=0)
    Ip_t[int(T_init/2):,:] = 0
    Ig_t[int(T_init/2):,:] = 0
    u_HPC, u_grid, I_mec = bm.for_loop(initial_net, (Ip_t, Ig_t), progress_bar=True)

    # decoding dynamics
    indices_est = np.arange(total_brainpy_iteration)
    activation_p_repeat = alpha_p_infer*np.repeat(activation_p[np.newaxis, :], total_brainpy_iteration, axis=0)
    activation_gs_repeat = params_PSC['alpha_g']*np.repeat(activation_gs[np.newaxis, :, :], total_brainpy_iteration, axis=0)
    u_HPC, u_grid, I_mec, z_record, phi_record = bm.for_loop(run_net, (indices_est, activation_p_repeat, activation_gs_repeat), progress_bar=True)

    return z_record, phi_record, u_HPC, u_grid, I_mec

def LPSC_GOP_decoder(LSCModel, PSCModel,total_iterations, z_t, phi_t, activation_p, activation_gs,alpha_p_infer):  
    '''
    GOP: Gradient based Optimization of Posterior
    activaiton_p shape [n_p]
    activation_gs shape [M, n_g]
    '''
    z_ts = []
    phi_ts = []
    z_ts.append(z_t.detach().cpu().clone())
    phi_ts.append(phi_t.detach().cpu().clone())

    sigma_p_infer = np.sqrt(np.sqrt(math.pi)*params_LSC['A_p']**3*params_LSC['rho_p']*params_LSC['tau_p']
                            /(params_LSC['a_p']*alpha_p_infer)) * params_prob['sigma_ratio']
    sigma_p_infer = torch.tensor(sigma_p_infer).to(device)
    
    for iteration in range(total_iterations):
        ## compute the log likelihood
        # partial ln P(rg|phi) / partial phi
        fg_prime = PSCModel.forward_modules_prime(phi_t) # shape [M, n_g]
        Ig_fgprime_prod = activation_gs * fg_prime # shape [M, n_g]
        Ig_fgprime_prod = torch.sum(Ig_fgprime_prod, dim=1) # shape [M]
        dphi_fr = Ig_fgprime_prod /  params_PSC['sigma_g']**2 
        dphi_fr = dphi_fr.to(device)

        # partial ln P(rp|z) / partial z
        fp_prime = LSCModel.forward_prime(z_t) # shape [n_p]
        Ip_fp_prime_prod = activation_p * fp_prime # shape [n_p]
        Ip_fp_prime_prod = torch.sum(Ip_fp_prime_prod, dim=0) # shape [1]
        dr_fr = Ip_fp_prime_prod / sigma_p_infer**2
        dr_fr = dr_fr.to(device)
        
        ## transition model
        phi_z = position2phase_modules(z_t, params_PSC)
        dis_phi = circ_dis(phi_z, phi_t) # shape [M]
        # partial ln P(phi|z) / partial phi
        dphi_tr = 1 / params_PSC['sigma_phi']**2 * dis_phi # shape [M]
        dphi_tr = dphi_tr.to(device)
        # partial ln P(phi|z) / partial z
        dr_tr = torch.sum(-params_PSC['Lphase'] / (params_PSC['lambda_gs'] * params_PSC['sigma_phi']**2) * dis_phi, dim=0) # shape [1]
        dr_tr = dr_tr.to(device)

        ## update
        dphi = dphi_fr + dphi_tr
        phi_t = phi_t + params_prob['eta'] * dphi
        # boundary condition
        # phi_t = torch.where(phi_t >= params_PSC['Lphase'], phi_t - params_PSC['Lphase'], phi_t)
        # phi_t = torch.where(phi_t < 0, phi_t + params_PSC['Lphase'], phi_t)
        phi_t = phi_t % params_PSC['Lphase']
        
        dr = dr_fr + dr_tr
        z_t = z_t + params_prob['eta'] * dr

        ## record
        z_ts.append(z_t.detach().cpu().clone())
        phi_ts.append(phi_t.detach().cpu().clone())
    return z_ts, phi_ts

def test_func():
    print(params_prob['eta'])

def run(args):
    torch.manual_seed(args.seed)
    logging_path = os.path.join('./results/',  args.exp_type)
    os.makedirs(logging_path, exist_ok=True)
    notebook = Logger(logging_path, f"decoding.csv")
    
    ### initialize LSC and PSC
    LSCModel = LSC(params_LSC).to(device)
    PSCModel = PSC(params_PSC).to(device)

    alpha_ps = np.linspace(0.01, 0.1, 6)
    for j, alpha_p in enumerate(alpha_ps):
        alpha_p_infer = float(alpha_p)
        for i in range(args.num_simulation):
            print('Simulation trial:', i)
            decoding_dict = dict() # keys: pos, phi
            ### stimulus and ground truth position
            x_g = 62.1
            x_p = 61.9
            x_gt = (x_g + x_p) / 2
            pos_gt = torch.tensor(x_gt, dtype=torch.float32)  # ground truth position
            pos_gt = pos_gt.to(device)
            phi_gt = position2phase_modules(pos_gt, params_PSC)
            
            pos_gt_np = pos_gt.detach().cpu().clone().numpy()
            phi_gt_np = phi_gt.detach().cpu().clone().numpy()

            z_t = pos_gt 
            phi_t = position2phase_modules(z_t, params_PSC)
            z_t_np = z_t.detach().clone().numpy()
            phi_t_np = phi_t.detach().clone().numpy()

            pos_g = torch.tensor(x_g, dtype=torch.float32)  # ground truth position
            pos_p = torch.tensor(x_p, dtype=torch.float32)  # ground truth position
            fp_t = LSCModel.forward(pos_p, noiseFlag=False).detach().cpu().clone().numpy()
            fg_t = PSCModel.forward(pos_g, noiseFlag=False).detach().cpu().clone().numpy()
            
            ### generate observation
            activation_p = LSCModel.forward(pos_p,noiseFlag=True)
            activation_gs = PSCModel.forward(pos_g, noiseFlag=True)
            
            ### NET: Network decoding method
            # transfer the activation to make it fit for brainpy
            activation_p_np = activation_p.detach().cpu().clone().numpy()
            activation_gs_np = activation_gs.detach().cpu().clone().numpy()

            ### initialize the brainpy network
            ## Place cells
            P_CANN = Place_net(z_min=0, z_max=params_LSC['L'], num=params_LSC['n_p'], 
                            a_p=params_LSC['a_p'], k=params_LSC['k_p'], tau=params_LSC['tau_p'], J0=params_LSC['J_p'])
            ## Grid cells
            G_CANNs = bm.NodeList()
            for i in range(params_PSC['M']):
                G_CANNs.append(Grid_net(L = params_PSC['lambda_gs_np'][i], z_min=0, z_max=params_LSC['L'], 
                                        num = params_PSC['n_g'], num_hpc=params_LSC['n_p'], a_g = params_PSC['a_gs_np'][i], 
                                        k_mec=params_PSC['k_gs'][i], tau=params_PSC['tau_gs'][i], J0=params_PSC['J_g'], W0=params_PSC['J_pg'][i]))
            ## Coupled Network
            Coupled_model = Coupled_Net(HPC_model=P_CANN, MEC_model_list=G_CANNs, num_module=params_PSC['M'])

            ### NET: Network decoding method
            z_record_net, phi_record_net, u_HPC, u_grids, I_mec = NET_decoder(Coupled_model, args.total_brainpy_iteration, z_t_np, phi_t_np, fp_t, fg_t, activation_p_np, activation_gs_np, alpha_p_infer)
            z_record_net = np.array(z_record_net)
            phi_record_net = np.array(phi_record_net) # shape [total_brainpy_iteration+1, M]
            u_HPC = np.array(u_HPC) # T x num_p
            u_grids = np.array(u_grids) # M x T x num_g
            I_mec = np.array(I_mec) # T x num_g

            ### GOP decoding method
            z_record_GOP, phi_record_GOP = LPSC_GOP_decoder(LSCModel, PSCModel, args.total_iterations, z_t, phi_t, activation_p, activation_gs, alpha_p_infer)
            z_record_GOP = np.array(z_record_GOP) # shape [total_iterations+1, 1]
            phi_record_GOP = np.array(phi_record_GOP) # shape [total_iterations+1, M]

            print(f"Ground truth position: {pos_gt_np}, Ground truth phase: {phi_gt_np}")
            print(f"NET estimated position: {z_record_net[-1]}, NET estimated phase: {phi_record_net[-1]}")
            print(f"GOP estimated position: {z_record_GOP[-1]}, GOP estimated phase: {phi_record_GOP[-1]}")

            z_est_net = z_record_net[-1][0]
            phi_est_net = phi_record_net[-1]
            z_est_GOP = z_record_GOP[-1]
            phi_est_GOP = phi_record_GOP[-1]
            decoding_dict[f'pos_GOP_{j}'] = z_est_GOP
            decoding_dict[f'pos_NET_{j}'] = z_est_net
            for module in range(params_PSC['M']):
                decoding_dict[f'phi{module+1}_GOP_{j}'] = phi_est_GOP[module]
                decoding_dict[f'phi{module+1}_NET_{j}'] = phi_est_net[module]
            # Save results
            notebook.log(decoding_dict)

if __name__ == "__main__":    
    print('Running inference')
    start_time = time.time()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_type', '-e', type=str, required=True, help="Results will be save in ./results/$exp_type, types are LSC_MAP, PSC_MAP, PSC_GO, LSC_PSC_GO, LSC_PSC_MAP")
    arg_parser.add_argument('--seed', '-s', type=int, default=1, help="random seed number")
    arg_parser.add_argument('--total_brainpy_iteration', '-b', type=int, default=5000, help="total iteration number for NET based on brainpy")
    arg_parser.add_argument('--total_iterations', '-t', type=int, default=5000, help="total iteration number for GOP")
    arg_parser.add_argument('--num_simulation', '-n', type=int, default=20, help="simulation number")
    args = arg_parser.parse_args()
    run(args)
    end_time = time.time()
    print('Total time:', end_time - start_time)