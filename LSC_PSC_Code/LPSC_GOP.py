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

from utils import metric
from utils.logger import Logger
from utils.helperFunctions import circ_dis, plot_z_decoding, plot_phi_decoding
from utils.params import params_prob, params_LSC, params_PSC
from utils.generation import LSC, PSC
from utils.ProbModel import position2phase_one_module, position2phase_modules

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LPSC_GOP_decoder(LSCModel, PSCModel,total_iterations, z_t, phi_t, activation_p, activation_gs):  
    '''
    GOP: Gradient based Optimization of Posterior
    activaiton_p shape [n_p]
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

        # partial ln P(rp|z) / partial z
        fp_prime = LSCModel.forward_prime(z_t) # shape [n_p]
        Ip_fp_prime_prod = activation_p * fp_prime # shape [n_p]
        Ip_fp_prime_prod = torch.sum(Ip_fp_prime_prod, dim=0) # shape [1]
        dr_fr = Ip_fp_prime_prod / params_LSC['sigma_p']**2
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

def run(args):
    torch.manual_seed(args.seed)
    logging_path = os.path.join('./results/',  args.exp_type)
    os.makedirs(logging_path, exist_ok=True)

    ### stimulus and ground truth position
    pos_gt = torch.tensor([101.5], dtype=torch.float32)  # ground truth position
    pos_gt = pos_gt.to(device)
    phi_gt = position2phase_modules(pos_gt, params_PSC)
    ### initialize LSC and PSC
    LSCModel = LSC(params_LSC).to(device)
    PSCModel = PSC(params_PSC).to(device)

    ### generate observation
    activation_p = LSCModel.forward(pos_gt,noiseFlag=True)
    activation_gs = PSCModel.forward(pos_gt, noiseFlag=True)

    z_t = pos_gt - params_prob["v"] * params_prob["dt"] # previous position
    # z_t = z_t.to(device)
    phi_t = position2phase_modules(z_t, params_PSC)
    
    pos_gt_record = pos_gt.detach().cpu().clone().numpy()
    phi_gt_record = phi_gt.detach().cpu().clone().numpy()
    z_init_record = z_t.detach().cpu().clone().numpy()
    phi_init_record = phi_t.detach().cpu().clone().numpy()

    z_ts, phi_ts = LPSC_GOP_decoder(LSCModel, PSCModel, args.total_iterations, z_t, phi_t, activation_p, activation_gs)

    z_ts = torch.stack(z_ts).detach().cpu().numpy() # shape [total_iterations+1, 1]
    phi_ts = torch.stack(phi_ts).detach().cpu().numpy() # shape [total_iterations+1, M]

    ### plot
    savepath = logging_path
    os.makedirs(savepath, exist_ok=True)
    plot_z_decoding(z_ts, pos_gt_record, savepath)
    plot_phi_decoding(phi_ts, phi_gt_record, params_PSC['M'], savepath)


if __name__ == "__main__":    
    print('Running inference')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_type', '-e', type=str, required=True, help="Results will be save in ./results/$exp_type, types are LSC_MAP, PSC_MAP, PSC_GO, LSC_PSC_GO, LSC_PSC_MAP")
    arg_parser.add_argument('--seed', '-s', type=int, default=1, help="random seed number")
    arg_parser.add_argument('--total_iterations', '-t', type=int, default=5000, help="total iteration number for training")
    args = arg_parser.parse_args()
    run(args)