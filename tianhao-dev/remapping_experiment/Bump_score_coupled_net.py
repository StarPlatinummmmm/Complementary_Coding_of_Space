import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from Network_Multiple_Maps import Place_net, Grid_net

z_min = 0
z_max = 20
place_num = 800
grid_num = 20
module_num = 7
a_p = 0.5
Spacing = np.linspace(6,20,module_num)
neuron_num = int(place_num)
num_map = 10
map_num_all = np.arange(num_map)+2
bump_score_orginal = np.zeros(num_map)
bump_score_others = np.zeros(num_map)
bump_score_diff = np.zeros(num_map)
bump_score_orginal_std = np.zeros(num_map)
bump_score_others_std = np.zeros(num_map)
bump_score_diff_std = np.zeros(num_map)
simulaiton_num = 2

def cosine_similarity(A, B):
    # 将输入转化为NumPy数组
    A = np.array(A)
    B = np.array(B)
    # 计算向量的点积
    dot_product = np.dot(A, B)
    # 计算向量的范数（即向量的长度）
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # 计算并返回余弦相似度
    cosine_sim = dot_product / (norm_A * norm_B)
    return cosine_sim

# map_num = 10
for i in range(num_map):
    map_num = map_num_all[i]
    bump_score_orginal_monte = np.zeros(simulaiton_num)
    bump_score_others_monte = np.zeros(simulaiton_num)
    bump_score_diff_monte = np.zeros(simulaiton_num)
    for j in range(simulaiton_num):
        Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=place_num, place_num=place_num, noise_stre=0.5)
        maps = bm.as_numpy(Place_cell.map) 
        place_index = bm.as_numpy(Place_cell.place_index) 


        Gird_module_list = bm.NodeList([])
        for module in range(module_num):
            Grid_cell = Grid_net(L = Spacing[module], maps=maps, place_index=place_index, neuron_num=grid_num, J0=5, a_g=a_p/Spacing[module]*2*bm.pi)
            Gird_module_list.append(Grid_cell)
            
        conn_out = Grid_cell.conn_out

        map_index = 0
        def run_net(indices, loc, input_stre):            
            place_r = Place_cell.r
            output = bm.zeros(place_num,)
            for Grid_cell in Gird_module_list:
                Grid_cell.step_run(indices, r_hpc= place_r, loc=loc, input_stre=input_stre,map_index = map_index, )
                output += Grid_cell.output
            Place_cell.step_run(indices, loc=loc, map_index = map_index, input_stre=input_stre, input_g = output)

        total_time = 10000
        start_time = 1000
        indices = bm.arange(total_time)
        loc = bm.zeros(total_time) + (z_max+z_min)/2
        input_stre = bm.zeros(total_time) 
        input_stre[:start_time] = 1.
        bm.for_loop(run_net, (indices, loc, input_stre), progress_bar = False)

        u = Place_cell.r
        loc_num = 100
        loc_candidate = np.linspace(z_min,z_max,loc_num,endpoint=False)
        bump_score = np.zeros(map_num,)
        max_score_pos = np.zeros(map_num,)
        for map_index in range(map_num):
            u_place = bm.as_numpy(u[place_index[map_index]])
            score_candidate = np.zeros(loc_num,)
            for loc_j in range(loc_num):
                bump = Place_cell.get_bump(map_index, loc_candidate[loc_j])
                u_place = u_place/bm.sum(u_place)
                score_candidate[loc_j] = bm.sum(bump*u_place)
            bump_score[map_index] = np.max(score_candidate)
            max_score_pos[map_index] = loc_candidate[np.argmax(score_candidate)]



        bump_score_orginal_monte[j] = bump_score[0]
        bump_score_others_monte[j] = np.max(bump_score[1:])
        bump_score_diff_monte[j] = bump_score_orginal_monte[j] - bump_score_others_monte[j]
        print((i*simulaiton_num+j)/(simulaiton_num*num_map))

    bump_score_orginal[i] = np.mean(bump_score_orginal_monte)
    bump_score_others[i] = np.mean(bump_score_others_monte)
    bump_score_diff[i] = np.mean(bump_score_diff_monte)

    bump_score_orginal_std[i] = np.std(bump_score_orginal_monte)
    bump_score_others_std[i] = np.std(bump_score_others_monte)
    bump_score_diff_std[i] = np.std(bump_score_diff_monte)


# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8))

# First subplot: bump_score_orginal and bump_score_others
ax1.errorbar(map_num_all, bump_score_orginal, yerr=bump_score_orginal_std, fmt='-o', label='Original')
ax1.errorbar(map_num_all, bump_score_others, yerr=bump_score_others_std, fmt='-o', label='Others')
ax1.set_title('Error Bar Plot of Bump Scores: Original and Others')
ax1.set_xlabel('Embedded Map Number')
ax1.set_ylabel('Bump Score')
ax1.legend()
ax1.grid(True)

# Second subplot: bump_score_diff
ax2.errorbar(map_num_all, bump_score_diff, yerr=bump_score_diff_std, fmt='-o', label='Difference')
ax2.set_title('Error Bar Plot of Bump Score Difference')
ax1.set_xlabel('Embedded Map Number')
ax2.set_ylabel('Bump Score Difference')
ax2.legend()
ax2.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig('bump_score_errorbars_coupled_net.png')

# Save the data
np.savez('bump_scores_coupled_net.npz',
         bump_score_orginal=bump_score_orginal,
         bump_score_others=bump_score_others,
         bump_score_diff=bump_score_diff,
         bump_score_orginal_std=bump_score_orginal_std,
         bump_score_others_std=bump_score_others_std,
         bump_score_diff_std=bump_score_diff_std)

plt.show()

