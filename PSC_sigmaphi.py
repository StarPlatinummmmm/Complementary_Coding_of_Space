'''
Probability model
x, phi = argmax_{x, phi} [ln P(z_t, phi | I_g)]
       = argmax_{x, phi} [ln P(I_g | phi) + ln P(phi | z_t)]

Using Gradient based optimization to find the MAP estimation
'''

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
from utils.helperFunctions import circ_dis, combined_sum_probabilities
from utils.params import params_prob, params_LSC, params_PSC
from utils.generation import LSC, PSC
from utils.Network import Place_net, Grid_net, Coupled_Net
from utils.ProbModel import position2phase_modules, position2phase_modules_batch
from utils.ProbModel import position2phase_loglikelihood_modules_batch_MAP, PSC_fr_loglikelihood_modules_batch_MAP

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

### GOP decoding
def PSC_GOP_decoder(PSCModel,total_iterations, z_t, phi_t, activation_gs):  
    '''
    GOP: Gradient based Optimization of Posterior
    activation_gs shape [M, n_g]
    '''
    z_ts = []
    phi_ts = []
    z_ts.append(z_t.detach().cpu().clone())
    phi_ts.append(phi_t.detach().cpu().clone())
    for iteration in range(total_iterations):
        ## compute the log likelihood
        # partial ln P(rg|phi) / partial phi
        fg_prime = PSCModel.forward_modules_prime(phi_t) # shape [M, n_g]
        Ig_fgprime_prod = activation_gs * fg_prime # shape [M, n_g]
        Ig_fgprime_prod = torch.sum(Ig_fgprime_prod, dim=1) # shape [M]

        dphi_fr = Ig_fgprime_prod /  params_PSC['sigma_g']**2 
        dphi_fr = dphi_fr.to(device)
        
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
        phi_t = phi_t % params_PSC['Lphase']
        z_t = z_t + params_prob['eta'] * dr_tr

        ## record
        if iteration % 100 == 0 or iteration == total_iterations - 1:
            z_ts.append(z_t.detach().cpu().clone())
            phi_ts.append(phi_t.detach().cpu().clone())
    return z_ts, phi_ts

### MAP decoding
def read_MAP_indices_max_vec(log_posterior):
    '''
    log_posterior shape [n_pos, n_phi, M]
    Note: we can compress log_posterior from [n_pos, n_phi, n_phi, ..., n_phi] to [n_pos, n_phi, M] 
            because different modules are independent, covariance matrix is diagonal, 
            non-diagonal combinations are abandoned.
    '''
    n_pos, n_phi, M = log_posterior.shape
    # Calculate the maximum probability values and their indices for each [n_phi, M] matrix
    max_prob_values = np.max(log_posterior, axis=1)
    max_prob_indices = np.argmax(log_posterior, axis=1)
    # Sum the maximum probability values across the M modules for each position
    max_prob_sums = np.sum(max_prob_values, axis=1)
    # Find the position with the highest sum of maximum probabilities
    z_est_index = np.argmax(max_prob_sums)
    max_log_posterior = max_prob_sums[z_est_index]
    phi_est_index = max_prob_indices[z_est_index]
    return z_est_index, phi_est_index

def PSC_MAP_decoder(LSCModel, PSCModel, activation_gs):  
    '''
    MAP: Maximum A Posteriori
    activation_gs shape [M, n_g]
    '''
    ## parameter space
    z_candidates = LSCModel.mu_p
    phi_candidates = PSCModel.mu_g # the density for different modules is the same
    phi_candidates_modules = phi_candidates[:,None].expand(-1,params_PSC['M']) # shape [n_phi, M]
    phi_z_candidates = position2phase_modules_batch(z_candidates, params_PSC) # shape [n_p, M]
    ## transition likelihood
    log_likelihood_tr = position2phase_loglikelihood_modules_batch_MAP(phi_candidates_modules, phi_z_candidates) # shape [n_pos, n_phi, M]
    log_likelihood_tr = log_likelihood_tr.to(device)
    ## firing rate likelihood
    fg_modules = PSCModel.forward_modules_batch(phi_candidates_modules) # shape [n_phi, M, n_g]
    log_likelihood_fr = PSC_fr_loglikelihood_modules_batch_MAP(activation_gs, fg_modules) # shape [n_phi, M]
    log_likelihood_fr = log_likelihood_fr.to(device)
    ## posterior
    log_posterior = log_likelihood_tr + log_likelihood_fr # shape [n_pos, n_phi, M]
    # to numpy
    log_posterior = log_posterior.detach().cpu().clone().numpy()
    ## MAP estimation
    z_est_index, phi_est_index = read_MAP_indices_max_vec(log_posterior)
    phi_est_index = np.array(phi_est_index)
    z_est = z_candidates[z_est_index]
    phi_est = phi_candidates[phi_est_index]
    z_est = z_est.detach().cpu().clone().numpy()
    phi_est = phi_est.detach().cpu().clone().numpy()
    return z_est, phi_est

### NET decoding

def NET_decoder(Coupled_model, total_brainpy_iteration, z_t, phi_t, fp_t, fg_t, activation_p, activation_gs,T_init = 800):
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
    activation_p_repeat = 0*np.repeat(activation_p[np.newaxis, :], total_brainpy_iteration, axis=0)
    activation_gs_repeat = params_PSC['alpha_g']*np.repeat(activation_gs[np.newaxis, :, :], total_brainpy_iteration, axis=0)
    u_HPC, u_grid, I_mec, z_record, phi_record = bm.for_loop(run_net, (indices_est, activation_p_repeat, activation_gs_repeat), progress_bar=True)

    return z_record, phi_record, u_HPC, u_grid, I_mec

def run(args):
    torch.manual_seed(args.seed)
    save_path = os.path.join('./results/',  args.exp_type)
    os.makedirs(save_path, exist_ok=True)

    ### initialize LSC and PSC
    LSCModel = LSC(params_LSC).to(device)
    PSCModel = PSC(params_PSC).to(device)

    ### stimulus and ground truth position
    pos_gt = torch.tensor(99.5, dtype=torch.float32)  # ground truth position
    pos_gt = pos_gt.to(device)
    phi_gt = position2phase_modules(pos_gt, params_PSC)

    pi_ampls = np.array([0.5, 1.0, 1.5, 2.0, 2.4, 2.8, 3.4, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    n_pi_ampls = len(pi_ampls)
    n_methods = 3
    z_est_arrays = np.zeros([n_pi_ampls, args.num_simulation, n_methods])
    # print('z_est_arrays shape:', z_est_arrays.shape)
    phi_est_arrays = np.zeros([n_pi_ampls, args.num_simulation, n_methods, params_PSC['M']])
    
    for pi_index, pi_ampl in enumerate(pi_ampls):
        # print('pi_index:', pi_index, 'pi_ampl:', pi_ampl)
        pi_ampl = float(pi_ampl)
        for n_index in range(args.num_simulation):
            print('Simulation trial:', n_index)
            ### generate observation
            activation_p = LSCModel.forward(pos_gt,noiseFlag=True)
            activation_gs = PSCModel.forward(pos_gt, noiseFlag=True, pi_ampl=pi_ampl)

            ### GOP decoding
            z_t = pos_gt - params_prob["v"] * params_prob["dt"] # previous position
            phi_t = position2phase_modules(z_t, params_PSC)
            z_t_np = z_t.detach().clone().numpy()
            phi_t_np = phi_t.detach().clone().numpy()

            z_ts, phi_ts = PSC_GOP_decoder(PSCModel, args.total_iterations, z_t, phi_t, activation_gs)
            z_ts = torch.stack(z_ts).detach().cpu().numpy() # shape [total_iterations+1, 1]
            phi_ts = torch.stack(phi_ts).detach().cpu().numpy() # shape [total_iterations+1, M]
            
            z_est_GOP = z_ts[-1]
            phi_est_GOP = phi_ts[-1]
            # clear z_ts, phi_ts
            z_ts, phi_ts = [], []

            pos_gt_np = pos_gt.detach().cpu().clone().numpy()
            phi_gt_np = phi_gt.detach().cpu().clone().numpy()

            fp_t = LSCModel.forward(z_t, noiseFlag=False).detach().cpu().clone().numpy()
            fg_t = PSCModel.forward(z_t, noiseFlag=False).detach().cpu().clone().numpy()

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
            z_record_net, phi_record_net, _, _, _ = NET_decoder(Coupled_model, args.total_brainpy_iteration, z_t_np, phi_t_np, fp_t, fg_t, activation_p_np, activation_gs_np)
            z_record_net = np.array(z_record_net)
            phi_record_net = np.array(phi_record_net) # shape [total_brainpy_iteration+1, M]
            z_est_net = z_record_net[-1][0]
            phi_est_net = phi_record_net[-1]
            # clear z_record_net, phi_record_net
            z_record_net, phi_record_net = [], []

            ### MAP decoding
            z_est_MAP, phi_est_MAP = PSC_MAP_decoder(LSCModel, PSCModel, activation_gs)

            # save the results
            z_est_arrays[pi_index,n_index,0] = z_est_GOP
            z_est_arrays[pi_index,n_index,1] = z_est_net
            z_est_arrays[pi_index,n_index,2] = z_est_MAP
            print('z_est:', z_est_GOP, z_est_net, z_est_MAP)

            for module in range(params_PSC['M']):
                phi_est_arrays[pi_index,n_index,0,module] = phi_est_GOP[module]
                phi_est_arrays[pi_index,n_index,1,module] = phi_est_net[module]
                phi_est_arrays[pi_index,n_index,2,module] = phi_est_MAP[module]

    # save the results
    np.save(os.path.join(save_path, 'z_est_arrays.npy'), z_est_arrays)
    np.save(os.path.join(save_path, 'phi_est_arrays.npy'), phi_est_arrays)


    return

if __name__ == "__main__":    
    print('Running inference')
    # start_time = time.time()
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--exp_type', '-e', type=str, required=True, help="Results will be save in ./results/$exp_type, types are LSC_MAP, PSC_MAP, PSC_GO, LSC_PSC_GO, LSC_PSC_MAP, All")
    # arg_parser.add_argument('--seed', '-s', type=int, default=1, help="random seed number")
    # arg_parser.add_argument('--total_brainpy_iteration', '-b', type=int, default=5000, help="total iteration number for NET based on brainpy")
    # arg_parser.add_argument('--total_iterations', '-t', type=int, default=5000, help="total iteration number for GOP")
    # arg_parser.add_argument('--num_simulation', '-n', type=int, default=60, help="simulation number")
    # args = arg_parser.parse_args()
    # run(args)
    # print('Time cost:', time.time()-start_time)

    print(params_PSC['sigma_phi'])