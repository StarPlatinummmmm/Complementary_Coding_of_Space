
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import itertools
from matplotlib.animation import FuncAnimation

def circ_dis(phi_1,phi_2):
    d = phi_1 - phi_2
    d = torch.where(d > math.pi, d - 2 * math.pi, d)
    d = torch.where(d < -math.pi, d + 2 * math.pi, d)
    return d

def circ_dis_np(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis[dis > np.pi] -= 2 * np.pi
    dis[dis < -np.pi] += 2 * np.pi
    return dis

def generate_stim_array(L=400, k=20, a=98.5):
    # Create an array of length L filled with zeros
    arr = np.zeros(L)
    # Fill the array according to the specified pattern
    for i in range(0, L, k):
        arr[i:i+k] = a + (i // k) * 0.25
    return arr

def combined_sum_probabilities(prob_matrix):
    n, m = prob_matrix.shape
    # Generate all possible index combinations where each element comes from a different module
    index_combinations = list(itertools.product(range(n), repeat=m))
    
    # Calculate the sum of probabilities for each combination
    prob_sums = []
    for combination in index_combinations:
        prob_sum = sum(prob_matrix[combination[i], i] for i in range(m))
        prob_sums.append(prob_sum)
    
    prob_sums = np.array(prob_sums)
    
    # Find the maximum probability sum and its corresponding indices
    max_prob_sum = np.max(prob_sums)
    max_index = np.argmax(prob_sums)
    max_combination = index_combinations[max_index]
    
    # Convert the max_combination to the required format [candidate_index, module_index]
    # max_combination_indices = [(max_combination[i], i) for i in range(m)]
    max_combination_indices = [max_combination[i] for i in range(m)]
    
    return prob_sums, max_prob_sum, max_combination_indices

def get_max_probabilities(prob_matrix):
    # Find the maximum probability value for each column
    max_prob_values = np.max(prob_matrix, axis=0)
    # Find the index of the maximum probability value for each column
    max_prob_indices = np.argmax(prob_matrix, axis=0)
    
    return max_prob_values, max_prob_indices

def plot_z_decoding(z_ts,z_gt,savepath,fontsize = 20,linewidth = 2.5):
    plt.figure(figsize=(20, 10))
    plt.plot(z_ts, linewidth=linewidth)
    T = len(z_ts)
    plt.plot([0, T], [z_gt, z_gt], 'r--', linewidth=linewidth)
    plt.xlabel('Iterations', fontsize=fontsize)
    plt.ylabel('location z', fontsize=fontsize)
    plt.gca().tick_params(width=linewidth)
    plt.gca().tick_params(labelsize=fontsize)
    min_z, max_z = min(z_ts), max(z_ts)
    plt.ylim([min_z-4, max_z+4])
    plt.tight_layout()
    filename = os.path.join(savepath, 'z_decoding.png')
    plt.savefig(filename)

def plot_phi_decoding(phi_ts,phi_gt,modules,savepath,fontsize = 20,linewidth = 2.5):
    # each cols plot at most 2 modules
    cols = 2
    if modules <= cols:
        rows = 1
    else:
        rows = math.ceil(modules / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
    for i in range(modules):
        if modules <= cols:
            ax = axs[i]
        else:
            ax = axs[i // cols, i % cols]
        T = len(phi_ts)
        ax.plot(phi_ts[:,i], linewidth=linewidth)
        ax.plot([0, T], [phi_gt[i], phi_gt[i]], '--', linewidth=linewidth)
        ax.set_xlabel('Iterations', fontsize=fontsize)
        ax.set_ylabel('Phase φ', fontsize=fontsize)
        ax.tick_params(width=linewidth)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylim([0, 2*np.pi])
    for i in range(modules, rows * cols):
        fig.delaxes(axs[i // cols, i % cols])
    filename = os.path.join(savepath, 'phi_decoding.png')
    plt.savefig(filename)
    
# def plot_network_animation(P_CANN, G_CANNs, u_HPC, u_grids, I_mec, Ig, n_step=50):
#     '''
#     u_grids: [M, T, N], u_HPC: [T, N], I_mec: [T, N]
#     '''
#     # each cols plot at most 2 modules
#     cols = 2
#     total_modules = len(G_CANNs)+1 # M+1
#     if total_modules <= 2: # one module
#         rows = 1
#     else:
#         rows = math.ceil(total_modules / cols)
#     T = u_HPC.shape[0]
#     u_HPC = u_HPC[::n_step, :]
#     u_grids = u_grids[:, ::n_step, :]
#     I_mec = I_mec[::n_step, :]

#     fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
#     for i in range(total_modules):
#         if total_modules <= cols:
#             ax = axs[i]
#         else:
#             ax = axs[i // cols, i % cols]
#         if i == 0: # place cell
#             ax.set_xlim(P_CANN.z_min, P_CANN.z_max)
#             ax.set_ylim(-0.01, 2.5)
#             line_up, = ax.plot([], [])
#             line_Imec, = ax.plot([], [])
#             ax.legend(['u_HPC', 'I_mec'])
#             ax.set_title('Population activities')
#         else:
#             ax.set_xlim(0, 2*np.pi)
#             ax.set_ylim(-0.01, 5)
#             line_ug, = ax.plot([], [])
#             # add legend for Ig
#             ax.plot(G_CANNs[i-1].x, Ig[i-1], label='Ig')
#             # show the legend
#             ax.legend(['u_grid', 'Ig'])
#             ax.set_title(f'Grid cell {i}')

#     def update(frame):
#         y_up = u_HPC[frame].flatten()
#         y_ug = u_grids[module][frame].flatten()
#         y_Imec = I_mec[frame].flatten()
#         line_up.set_data(P_CANN.x, y_up)
#         line_ug.set_data(G_CANNs[0].x, y_ug)
#         line_Imec.set_data(P_CANN.x, y_Imec)
#         return line_up, line_ug, line_Imec

#     ani = FuncAnimation(fig, update, frames=T, interval=20, blit=True)
#     plt.show()
def plot_network_animation(P_CANN, G_CANNs, u_HPC, u_grids, I_mec, Ig, n_step=50):
    '''
    u_grids: [M, T, N], u_HPC: [T, N], I_mec: [T, N]
    '''
    # each cols plot at most 2 modules
    cols = 2
    total_modules = len(G_CANNs) + 1 # M + 1
    if total_modules <= 2: # one module
        rows = 1
    else:
        rows = math.ceil(total_modules / cols)
        
    # Subsample the arrays
    u_HPC = u_HPC[::n_step, :]
    u_grids = u_grids[:, ::n_step, :]
    I_mec = I_mec[::n_step, :]
    
    # Determine the new number of frames after subsampling
    T = u_HPC.shape[0]

    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))

    lines = []  # To store line objects

    for i in range(total_modules):
        if total_modules <= cols:
            ax = axs[i]
        else:
            ax = axs[i // cols, i % cols]
        if i == 0: # place cell
            ax.set_xlim(P_CANN.z_min, P_CANN.z_max)
            ax.set_ylim(-0.01, 2.5)
            line_up, = ax.plot([], [])
            line_Imec, = ax.plot([], [])
            lines.append((line_up, line_Imec))
            ax.legend(['u_HPC', 'I_mec'])
            ax.set_title('Population activities')
        else:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(-0.01, 5)
            line_ug, = ax.plot([], [])
            # add legend for Ig
            ax.plot(G_CANNs[i - 1].x, Ig[i - 1], label='Ig')
            # show the legend
            ax.legend(['u_grid', 'Ig'])
            ax.set_title(f'Grid cell {i}')
            lines.append((line_ug,))

    def update(frame):
        for i in range(total_modules):
            if i == 0: # place cell
                y_up = u_HPC[frame].flatten()
                y_Imec = I_mec[frame].flatten()
                lines[i][0].set_data(P_CANN.x, y_up)
                lines[i][1].set_data(P_CANN.x, y_Imec)
            else:
                y_ug = u_grids[i - 1, frame].flatten()
                lines[i][0].set_data(G_CANNs[i - 1].x, y_ug)
        return [line for line_tuple in lines for line in line_tuple]

    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=True)
    plt.show()