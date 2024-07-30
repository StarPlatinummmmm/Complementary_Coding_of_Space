'''
Probability model
x, phi = argmax_{x, phi} [ln P(z_t, phi | I_g)]
       = argmax_{x, phi} [ln P(I_g | phi) + ln P(phi | z_t)]

Using Gradient based optimization to find the MAP estimation
'''

import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os,sys
# import multiprocessing as mp

from utils import metric
from utils.logger import Logger
from utils.params import params_prob, params_LSC, params_PSC
from utils.generation import LSC, PSC
from utils.ProbModel import position2phase_modules, position2phase_modules_batch
from utils.ProbModel import position2phase_loglikelihood_modules_batch_MAP, PSC_fr_loglikelihood_modules_batch_MAP
# from utils.ProbModel import position2phase_one_module_batch
# from utils.ProbModel import PSC_fr_loglikelihood_batch_MAP, position2phase_loglikelihood_batch_MAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def process_pos(pos_index, prob_matrix):
#     prob_sums, max_prob_sum, max_combination_indices = combined_sum_probabilities(prob_matrix[pos_index])
#     return max_prob_sum, pos_index, max_combination_indices

# def read_MAP_indices_parallel(log_posterior):
#     '''
#     log_posterior shape [n_pos, n_phi, M]
#     '''
#     n_pos, n_phi, M = log_posterior.shape
#     # Create a pool of processes
#     pool = mp.Pool(processes=mp.cpu_count())
#     # Distribute the work among the processes
#     results = pool.starmap(process_pos, [(i, log_posterior) for i in range(n_pos)])
#     # Close the pool and wait for the work to finish
#     pool.close()
#     pool.join()
#     # Find the maximum probability sum and its corresponding indices
#     max_log_posterior = -np.inf
#     z_est_index = -1.5
#     phi_est_index = np.ones(M) * -1.5
#     for max_prob_sum, pos_index, max_combination_indices in results:
#         if max_prob_sum > max_log_posterior:
#             max_log_posterior = max_prob_sum
#             z_est_index = pos_index
#             phi_est_index = max_combination_indices 
#     # phi_est_index = phi_est_index.astype(int)
#     return z_est_index, phi_est_index

# def read_MAP_indices(log_posterior):
#     '''
#     log_posterior shape [n_pos, n_phi, M]
#     Note: we can compress log_posterior from [n_pos, n_phi, n_phi, ..., n_phi] to [n_pos, n_phi, M] 
#             because different modules are independent, covariance matrix is diagonal, 
#             non-diagonal combinations are abandoned.
#     '''
#     n_pos, n_phi, M = log_posterior.shape
#     # initialize max_log_posterior as negative infinity
#     max_log_posterior = -np.inf
#     z_est_index = -1.5 # use fractional number make it not index-like for initialization
#     phi_est_index = np.ones(M) * -1.5 # use fractional number make it not index-like for initialization
#     for i in range(n_pos):
#         prob_sums, max_prob_sum, max_combination_indices = combined_sum_probabilities(log_posterior[i,:,:])
#         if max_prob_sum > max_log_posterior:
#             max_log_posterior = max_prob_sum
#             z_est_index = i
#             for j in range(M):
#                 phi_est_index[j] = max_combination_indices[j]
#     # phi_est_index = phi_est_index.astype(int)
#     return z_est_index, phi_est_index

# def read_MAP_indices_max(log_posterior):
#     '''
#     log_posterior shape [n_pos, n_phi, M]
#     Note: we can compress log_posterior from [n_pos, n_phi, n_phi, ..., n_phi] to [n_pos, n_phi, M] 
#             because different modules are independent, covariance matrix is diagonal, 
#             non-diagonal combinations are abandoned.
#     '''
#     n_pos, n_phi, M = log_posterior.shape
#     # initialize max_log_posterior as negative infinity
#     max_log_posterior = -np.inf
#     z_est_index = -1.5 # use fractional number make it not index-like for initialization
#     phi_est_index = np.ones(M) * -1.5 # use fractional number make it not index-like for initialization
#     for i in range(n_pos):
#         max_prob_values, max_prob_indices = get_max_probabilities(log_posterior[i,:,:])
#         max_prob_sum = np.sum(max_prob_values)
#         if max_prob_sum > max_log_posterior:
#             max_log_posterior = max_prob_sum
#             z_est_index = i
#             for j in range(M):
#                 phi_est_index[j] = max_prob_indices[j]
#     # phi_est_index = phi_est_index.astype(int)
#     return z_est_index, phi_est_index

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
    # z_est_index, phi_est_index = read_MAP_indices(log_posterior)
    # z_est_index, phi_est_index = read_MAP_indices_parallel(log_posterior)
    # z_est_index, phi_est_index = read_MAP_indices_max(log_posterior)
    z_est_index, phi_est_index = read_MAP_indices_max_vec(log_posterior)
    phi_est_index = np.array(phi_est_index)
    # phi_est_index = phi_est_index.astype(int)
    z_est = z_candidates[z_est_index]
    phi_est = phi_candidates[phi_est_index]
    z_est = z_est.detach().cpu().clone().numpy()
    phi_est = phi_est.detach().cpu().clone().numpy()
    return z_est, phi_est


# def PSC_brute_MAP_decoder(LSCModel, PSCModel, activation_gs):  
#     '''
#     MAP: Maximum A Posteriori
#     brute: more space for less running time, instead of [n_pos, n_phi, M], we use [n_pos, n_phi, n_phi, ..., n_phi]
#     activation_gs shape [M, n_g]
#     '''
#     ## parameter space
#     z_candidates = LSCModel.mu_p
#     phi_candidates = PSCModel.mu_g # the density for different modules is the same
#     # repeat phi_candidates to [n_phi, n_phi, ..., n_phi] -> phi_candidates_modules
#     n_pos = z_candidates.shape[0]
#     n_phi = phi_candidates.shape[0]
#     phi_candidates_modules = phi_candidates.view((n_phi,) + (1,) * (params_PSC['M']-1)).expand((n_phi,) * params_PSC['M'])    

#     for module in range(params_PSC['M']):
#         phi_z_candidates = position2phase_one_module_batch(z_candidates, module, params_PSC)
#         ## transition likelihood


def run(args):
    torch.manual_seed(args.seed)
    logging_path = os.path.join('./results/',  args.exp_type)
    os.makedirs(logging_path, exist_ok=True)
    # notebook = Logger(logging_path, f"decoding.csv")
    decoding_dict = dict() # keys: pos, phi

    ### stimulus and ground truth position
    pos_gt = torch.tensor([30.], dtype=torch.float32)  # ground truth position
    pos_gt = pos_gt.to(device)
    phi_gt = position2phase_modules(pos_gt, params_PSC)
    ### initialize LSC and PSC
    LSCModel = LSC(params_LSC).to(device)
    PSCModel = PSC(params_PSC).to(device)

    ### generate observation
    noiseFlag = True
    activation_p = LSCModel.forward(pos_gt,noiseFlag=noiseFlag)
    activation_gs = PSCModel.forward(pos_gt, noiseFlag=noiseFlag)

    z_est, phi_est = PSC_MAP_decoder(LSCModel, PSCModel, activation_gs)
    pos_gt_record = pos_gt.detach().cpu().clone().numpy()
    phi_gt_record = phi_gt.detach().cpu().clone().numpy()
    print('no noise' if noiseFlag == False else 'with noise')
    print(f"Groud truth position: {pos_gt_record}, Groud truth phase: {phi_gt_record}")
    print(f"Estimated position: {z_est}, Estimated phase: {phi_est}")



if __name__ == "__main__":    
    print('Running inference')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_type', '-e', type=str, required=True, help="Results will be save in ./results/$exp_type, types are LSC_MAP, PSC_MAP, PSC_GO, LSC_PSC_GO, LSC_PSC_MAP")
    arg_parser.add_argument('--seed', '-s', type=int, default=1, help="random seed number")
    arg_parser.add_argument('--total_iterations', '-t', type=int, default=2000, help="total iteration number for training")
    args = arg_parser.parse_args()
    run(args)