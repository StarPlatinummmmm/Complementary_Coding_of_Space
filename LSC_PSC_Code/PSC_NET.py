
import numpy as np
import math
import torch
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import argparse
import os,sys

from utils import metric
from utils.logger import Logger
from utils.helperFunctions import plot_z_decoding, plot_phi_decoding, plot_network_animation
from utils.params import params_prob, params_LSC, params_PSC
from utils.ProbModel import position2phase_one_module, position2phase_modules
from utils.Network import Place_net, Grid_net, Coupled_Net
from utils.generation import LSC, PSC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    activation_p_repeat = params_LSC['alpha_p']*np.repeat(activation_p[np.newaxis, :], total_brainpy_iteration, axis=0)
    activation_gs_repeat = params_PSC['alpha_g']*np.repeat(activation_gs[np.newaxis, :, :], total_brainpy_iteration, axis=0)
    u_HPC, u_grid, I_mec, z_record, phi_record = bm.for_loop(run_net, (indices_est, activation_p_repeat, activation_gs_repeat), progress_bar=True)

    return z_record, phi_record, u_HPC, u_grid, I_mec


def run(args):
    torch.manual_seed(args.seed)
    logging_path = os.path.join('./results/',  args.exp_type)
    os.makedirs(logging_path, exist_ok=True)
    # notebook = Logger(logging_path, f"decoding.csv")
    decoding_dict = dict() # keys: pos, phi

    ### stimulus and ground truth position
    pos_gt = torch.tensor([100.], dtype=torch.float32)  # ground truth position
    pos_gt = pos_gt.to(device)
    phi_gt = position2phase_modules(pos_gt, params_PSC)
    pos_gt_np = pos_gt.detach().cpu().clone().numpy()
    phi_gt_np = phi_gt.detach().cpu().clone().numpy()

    ### initialize LSC and PSC
    LSCModel = LSC(params_LSC).to(device)
    PSCModel = PSC(params_PSC).to(device)

    ### generate observation
    activation_p = LSCModel.forward(pos_gt,noiseFlag=True)
    activation_gs = PSCModel.forward(pos_gt, noiseFlag=True)
    z_t = pos_gt - params_prob["v"] * params_prob["dt"] # previous position
    phi_t = position2phase_modules(z_t, params_PSC)
    fp_t = LSCModel.forward(z_t, noiseFlag=False)
    fg_t = PSCModel.forward(z_t, noiseFlag=False)

    ### NET: Network decoding method
    # transfer the activation to make it fit for brainpy
    activation_p_np = activation_p.detach().clone().numpy()
    activation_gs_np = activation_gs.detach().clone().numpy()
    z_t_np = z_t.detach().clone().numpy()
    phi_t_np = phi_t.detach().clone().numpy()
    fp_t = fp_t.detach().clone().numpy()
    fg_t = fg_t.detach().clone().numpy()


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

    z_record, phi_record, u_HPC, u_grids, I_mec = NET_decoder(Coupled_model, args.total_brainpy_iteration, z_t_np, phi_t_np, fp_t, fg_t, activation_p_np, activation_gs_np)
    z_record = np.array(z_record)
    phi_record = np.array(phi_record)
    u_HPC = np.array(u_HPC) # T x num_p
    u_grids = np.array(u_grids) # M x T x num_g

    z_est = z_record[-1]
    phi_est = phi_record[-1,:]
    print(f"Ground truth position: {pos_gt_np}, Ground truth phase: {phi_gt_np}")
    print(f"Estimated position: {z_est}, Estimated phase: {phi_est}")

    # visualization
    plot_z_decoding(z_record,pos_gt_np,logging_path)
    plot_phi_decoding(phi_record,phi_gt_np,params_PSC['M'],logging_path)
    plot_network_animation(P_CANN, G_CANNs, u_HPC, u_grids, I_mec, activation_gs_np, n_step=50)

if __name__ == "__main__":    
    print('Running inference')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_type', '-e', type=str, required=True, help="Results will be save in ./results/$exp_type, types are LSC_MAP, PSC_MAP, PSC_GO, LSC_PSC_GO, LSC_PSC_MAP")
    arg_parser.add_argument('--seed', '-s', type=int, default=1, help="random seed number")
    arg_parser.add_argument('--total_iterations', '-t', type=int, default=5000, help="total iteration number for GOP")
    arg_parser.add_argument('--total_brainpy_iteration', '-b', type=int, default=5000, help="total iteration number for NET based on brainpy")
    args = arg_parser.parse_args()
    run(args)