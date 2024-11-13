
'''
Probability model
x, phi = argmax_{x, phi} [ln P(x_t, phi | I_g)]
       = argmax_{x, phi} [ln P(I_g | phi) + ln P(phi | x_t)]
'''

import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import time
import os, sys

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from params import params_prob, params_LSC, params_PSC
    from generation import LSC, PSC
else:
    from utils.params import params_prob, params_LSC, params_PSC
    from utils.generation import LSC, PSC
    
### Encoding
'''
Function name format:
Batch means candidates/templates for postion and phase
modules indicate parallel processing for different modules
n_pos, n_phi: batch size for position and phase
n_p, n_g: number of neurons for position and phase, corresponding to preferred positions 'mu' and phases 'theta'
'''

def LSC_loglikelihood_cth(Ip, fp):
    '''
    Ip shape [n_p], fp shape [n_pos, n_p]
    '''
    sigma_p = params_LSC['sigma_p']
    Ip_expand = Ip[None, :].expand(fp.shape[0], -1)
    log_prob = - Ip_expand* fp 
    # sum over the neuron dimension
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob # shape [n_pos]

def LSC_fr_loglikelihood_batch_MAP(Ip, fp):
    '''
    Ip shape [n_p], fp shape [n_pos, n_p]
    '''
    sigma_p = params_LSC['sigma_p']
    Ip_expand = Ip[None, :].expand(fp.shape[0], -1)
    log_prob = -(Ip_expand - fp)**2 / (2*sigma_p**2) # shape [n_pos, n_p]
    # sum over the neuron dimension
    log_prob = torch.sum(log_prob, dim=-1)
    log_prob = (log_prob - torch.min(log_prob))/(torch.max(log_prob)- torch.min(log_prob))
    return log_prob # shape [n_pos]

def LSC_fr_loglikelihood_GOP(Ip, fp):
    '''
    Ip shape [n_p], fp shape [n_p]
    '''
    sigma_p = params_LSC['sigma_p']
    log_prob = -(Ip - fp)**2 / (2*sigma_p**2) 
    # sum over the neuron dimension
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob # scalar

def PSC_fr_loglikelihood_batch_MAP(Ig, fg, module):
    '''
    Ig shape [n_g], fg shape [n_phi, n_g], for one module
    '''
    sigma_g = params_PSC['sigma_g'][module]
    Ig_expand = Ig[None, :].expand(fg.shape[0], -1)
    log_prob = -0.5 * (Ig_expand - fg)**2 / sigma_g**2 
    # sum over the neuron dimension
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob # shape [n_phi]

def PSC_fr_loglikelihood_modules_batch_MAP(Ig, fg):
    '''
    Ig shape [M, n_g], fg shape [n_phi, M, n_g]
    '''
    sigma_g = params_PSC['sigma_g'] # shape [M]
    # log_prob = -0.5 * (Ig - fg)**2 / sigma_g**2 
    # log_prob shape [n_phi, n_g]
    Ig_expand = Ig[None, :, :].expand(fg.shape[0], Ig.shape[0], -1) # shape [n_phi, M, n_g]
    log_prob = -0.5 * (Ig_expand - fg)**2 / sigma_g[None,:,None]**2 
    # sum over the neuron dimension
    log_prob = torch.sum(log_prob, dim=-1) 
    ### don't sum over the module dimension here, as M indicates n_phi for different modules
    return log_prob # shape [n_phi, M]

def PSC_fr_loglikelihood_GOP(Ig, fg):
    '''
    Ig shape [n_gs], fg shape [n_gs]
    we assume sigma_g is the same for all modules
    '''
    sigma_g = params_PSC['sigma_g'] 
    log_prob = -(Ig - fg)**2 / (2*sigma_g**2) 
    # sum over the neuron dimension
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob # scalar

def position2phase_one_module(x, module, params_PSC):
    Lphase = params_PSC['Lphase']
    lambda_g = params_PSC['lambda_gs'][module]
    phi = (x / lambda_g * Lphase) % Lphase
    return phi # scalar

def position2phase_one_module_batch(x, module,params_PSC):
    Lphase = params_PSC['Lphase']
    lambda_g = params_PSC['lambda_gs'][module]
    phi = (x[:, None] / lambda_g * Lphase) % Lphase
    return phi # [n_pos]

def position2phase_modules(x,params_PSC):
    Lphase = params_PSC['Lphase']
    lambda_gs = params_PSC['lambda_gs']
    phi = (x/lambda_gs * Lphase) % Lphase
    return phi # [M]

def position2phase_modules_batch(x,params_PSC):
    Lphase = params_PSC['Lphase']
    lambda_gs = params_PSC['lambda_gs']
    phi = (x[:, None]/lambda_gs * Lphase) % Lphase
    return phi # [n_pos, M]

def position2phase_loglikelihood_batch_MAP(phi, phi_x, module):
    '''
    P(phi | x), phi shape [n_phi], phi_x shape [n_pos], for one module
    '''
    sigma_phi = params_PSC['sigma_phi'][module]
    kappa_phi = 1 / (sigma_phi)**2
    phi_x_expand = phi_x[:, None].expand(-1,phi.shape[0])
    phi_expand = phi[None, :].expand(phi_x.shape[0], -1)
    log_prob = kappa_phi*torch.cos(phi_expand-phi_x_expand) 
    return log_prob # shape [n_pos, n_phi]

def position2phase_loglikelihood_modules_batch_MAP(phi, phi_x):
    '''
    P(phi | x), phi_x shape [n_pos,M],  phi shape [n_phi, M]
    '''
    sigma_phi = params_PSC['sigma_phi'] # shape [M]
    kappa_phi = 1 / (sigma_phi)**2
    phi_x_expand = phi_x[:, None, :].expand(phi_x.shape[0], phi.shape[0], -1)
    # phi_expand = phi[None, :, None].expand(phi_x.shape[0], phi.shape[0], phi_x.shape[1])
    phi_expand = phi[None, :, :].expand(phi_x.shape[0], phi.shape[0], -1)
    log_prob = kappa_phi*torch.cos(phi_expand-phi_x_expand) # shape [n_pos, n_phi, M]
    return log_prob # shape [n_pos, n_phi, M]

def position2phase_loglikelihood_GOP(phi, phi_x):
    '''
    P(phi | x), phi shape [M], phi_x is a scalar
    '''
    sigma_phi = params_PSC['sigma_phi']
    kappa_phi = 1 / (sigma_phi)**2
    log_prob = kappa_phi*torch.cos(phi-phi_x)
    return log_prob # shape [M]


def prior_function(phi, z_candidates):
    '''
    P(phi | x), phi shape [M], phi_x is a scalar
    '''
    phi_x = position2phase_modules_batch(z_candidates, params_PSC) # shape [n_pos, M]
    sigma_phi = params_PSC['sigma_phi']
    kappa_phi = 1 / (sigma_phi)**2
    log_prob = kappa_phi*torch.cos(phi-phi_x) # shape [n_pos, M]
    log_prob_z = torch.sum(log_prob, axis=1)
    return log_prob_z # shape [M]


if __name__ == "__main__":
    print('Example usage')
    # stimulus
    position = torch.tensor([params_prob["L"] // 3], dtype=torch.float32)  # Use tensor for position
    noiseFlag = True

    PSCModel = PSC(params_PSC)
    activation_gs = PSCModel.forward(position, noiseFlag)
    print(activation_gs.shape)
    
    module_idx = 0
    phi = position2phase_one_module(position, module_idx)
    activation_g = PSCModel.forward_one_module(phi, module_idx)
    print(activation_g.shape)

    phi = position2phase_modules(position, params_PSC)
    activation_gs = PSCModel.forward_modules(phi)
    print(activation_gs.shape)