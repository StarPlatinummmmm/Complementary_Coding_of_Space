'''
This script is used to compare the noise robustness of place cell and grid cell decoding methods.
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
from utils.helperFunctions import circ_dis
from utils.params import params_prob, params_LSC, params_PSC
from utils.generation import LSC, PSC
from utils.Network import Place_net, Grid_net, Coupled_Net
from utils.ProbModel import position2phase_modules, position2phase_modules_batch
from utils.ProbModel import position2phase_loglikelihood_modules_batch_MAP, PSC_fr_loglikelihood_modules_batch_MAP, LSC_fr_loglikelihood_batch_MAP

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

def PSC_MAP_decoder(PSCModel, activation_gs, n_pos=1000, n_phi=100):  
    '''
    MAP: Maximum A Posteriori
    activation_gs shape [M, n_g]
    '''
    L = params_prob['L']
    ## parameter space
    # z_candidates = LSCModel.mu_p
    # phi_candidates = PSCModel.mu_g # the density for different modules is the same
    z_candidates = np.linspace(0, L, n_pos, endpoint=False)
    phi_candidates = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    ### Q: candidate number is not necessarily equal with the neuron number?

    phi_candidates_modules = phi_candidates[:,None].expand(-1,params_PSC['M']) # shape [n_phi, M]
    phi_z_candidates = position2phase_modules_batch(z_candidates, params_PSC) # shape [n_pos, M]
    ## transition likelihood
    log_likelihood_tr = position2phase_loglikelihood_modules_batch_MAP(phi_candidates_modules, phi_z_candidates) # shape [n_pos, n_phi, M]
    log_likelihood_tr = log_likelihood_tr.to(device) ### Q: prior correlation p(z, \phi)?
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


def LSC_MAP_decoder(LSCModel, activation_p):  
    '''
    MAP: Maximum A Posteriori
    '''
    ## parameter space
    z_candidates = LSCModel.mu_p
    # MAP estimation
    max_index = torch.argmax(activation_p)
    z_est = z_candidates[max_index]
    z_est = z_est.detach().cpu().clone().numpy()
    return z_est

    # ## firing rate likelihood
    # fp = LSCModel.forward_batch(z_candidates) # shape [n_pos, n_p]
    # log_likelihood_fr = LSC_fr_loglikelihood_batch_MAP(activation_p, fp) # shape [n_pos]
    # log_posterior = log_likelihood_fr.to(device)
    # # to numpy
    # log_posterior = log_posterior.detach().cpu().clone().numpy()
    # ## MAP estimation
    # z_est_index = np.argmax(log_posterior)
    # z_est = z_candidates[z_est_index]
    # z_est = z_est.detach().cpu().clone().numpy()
    # return z_est

def run(args):
    torch.manual_seed(args.seed)
    save_path = os.path.join('./results/',  args.exp_type)
    os.makedirs(save_path, exist_ok=True)

    ### initialize LSC and PSC
    LSCModel = LSC(params_LSC).to(device)
    PSCModel = PSC(params_PSC).to(device)

    ### stimulus and ground truth position
    pos_gt = torch.tensor(30., dtype=torch.float32)  # ground truth position
    pos_gt = pos_gt.to(device)
    phi_gt = position2phase_modules(pos_gt, params_PSC)

    pos_gt_np = pos_gt.detach().cpu().clone().numpy()
    phi_gt_np = phi_gt.detach().cpu().clone().numpy()
    
    rate_ampls = np.array([0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 7., 8., 9., 10.])
    
    n_rate_ampls = len(rate_ampls)
    n_methods = 3
    z_est_arrays = np.zeros([n_rate_ampls, args.num_simulation, n_methods])
    phi_est_arrays = np.zeros([n_rate_ampls, args.num_simulation, n_methods, params_PSC['M']])
    
    for pi_index, rate_ampl in enumerate(rate_ampls):
        rate_ampl = float(rate_ampl)
        for n_index in range(args.num_simulation):
            print('Simulation trial:', n_index)
            ### generate observation
            activation_p = LSCModel.forward(pos_gt,noiseFlag=True,  rate_ampl=rate_ampl)
            activation_gs = PSCModel.forward(pos_gt, noiseFlag=True, rate_ampl=rate_ampl)

            ### PSC GOP decoding
            # z_t = pos_gt - params_prob["v"] * params_prob["dt"] # previous position
            z_t = pos_gt # previous position
            phi_t = position2phase_modules(z_t, params_PSC)
            z_ts, phi_ts = PSC_GOP_decoder(PSCModel, args.total_iterations, z_t, phi_t, activation_gs)
            z_ts = torch.stack(z_ts).detach().cpu().numpy() # shape [total_iterations+1, 1]
            phi_ts = torch.stack(phi_ts).detach().cpu().numpy() # shape [total_iterations+1, M]
            
            z_est_GOP = z_ts[-1]
            phi_est_GOP = phi_ts[-1]
            # clear z_ts, phi_ts
            z_ts, phi_ts = [], []

            ### PSC MAP decoding
            z_est_MAP, phi_est_MAP = PSC_MAP_decoder(LSCModel, PSCModel, activation_gs)

            ### LSC MAP decoding
            z_est_LSC = LSC_MAP_decoder(LSCModel, activation_p)

            # save the results
            z_est_arrays[pi_index,n_index,0] = z_est_GOP
            z_est_arrays[pi_index,n_index,1] = z_est_MAP
            z_est_arrays[pi_index,n_index,2] = z_est_LSC
            print('z_est:', z_est_GOP, z_est_MAP, z_est_LSC)

            for module in range(params_PSC['M']):
                phi_est_arrays[pi_index,n_index,0,module] = phi_est_GOP[module]
                phi_est_arrays[pi_index,n_index,1,module] = phi_est_MAP[module]
                phi_est_arrays[pi_index,n_index,2,module] = None

    # save the results
    np.save(os.path.join(save_path, 'z_est_arrays.npy'), z_est_arrays)
    np.save(os.path.join(save_path, 'phi_est_arrays.npy'), phi_est_arrays)

    return

if __name__ == "__main__":    
    print('Running inference')
    start_time = time.time()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_type', '-e', type=str, required=True, help="Results will be save in ./results/$exp_type, types are LSC_MAP, PSC_MAP, PSC_GO, LSC_PSC_GO, LSC_PSC_MAP, All")
    arg_parser.add_argument('--seed', '-s', type=int, default=1, help="random seed number")
    arg_parser.add_argument('--total_iterations', '-t', type=int, default=4000, help="total iteration number for GOP")
    arg_parser.add_argument('--num_simulation', '-n', type=int, default=25, help="simulation number")
    args = arg_parser.parse_args()
    run(args)
    print('Time cost:', time.time()-start_time)
