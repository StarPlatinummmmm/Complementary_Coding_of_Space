import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
import jax
from Network_Multiple_Maps import Place_net, Grid_net
import time
from tqdm import tqdm

# 记录开始时间
start_time = time.time()

# Define constants
z_min, z_max = 0, 20
module_num = 7
exp_num = 7
a_p = 0.5
Spacing = bm.linspace(6, 20, module_num)
num_map = 15
map_num_all = bm.arange(num_map) + 2 # 2-17
simulaiton_num = 100
place_num = np.linspace(600, 1000, exp_num, dtype=int)
grid_num = place_num // 40
max_map_num_coupled = np.zeros(exp_num)
max_map_num_only = np.zeros(exp_num)

for exp_i in range(exp_num):
    # Main loop
    for i, map_num in enumerate(map_num_all):
        bump_score_orginal_monte = bm.Variable(bm.zeros(simulaiton_num))
        bump_score_others_monte = bm.Variable(bm.zeros(simulaiton_num))
        bump_score_diff_monte = bm.Variable(bm.zeros(simulaiton_num))

        bump_score_orginal_monte_only = bm.Variable(bm.zeros(simulaiton_num))
        bump_score_others_monte_only = bm.Variable(bm.zeros(simulaiton_num))
        bump_score_diff_monte_only = bm.Variable(bm.zeros(simulaiton_num))

        # 更新函数
        def update_func(j):
            Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=place_num[exp_i], place_num=place_num[exp_i],
                                noise_stre=0.5)
            maps = Place_cell.map
            place_index = Place_cell.place_index

            # Grid module list initialization
            Gird_module_list = [Grid_net(L=Spacing[module], maps=maps, place_index=place_index, 
                                         neuron_num=grid_num[exp_i], J0=5,
                                         a_g=a_p / Spacing[module] * 2 * bm.pi) for module in range(module_num)]
            map_index = 0

            Place_cell_only = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=place_num[exp_i]+grid_num[exp_i]*module_num, place_num=place_num[exp_i]+grid_num[exp_i]*module_num, noise_stre=0.5)

            def run_net(indices, loc, input_stre):
                r_hpc = Place_cell.r
                output = bm.zeros(place_num[exp_i])
                for Grid_cell in Gird_module_list:
                    Grid_cell.step_run(indices, r_hpc=r_hpc, loc=loc, input_stre=input_stre, map_index=map_index)
                    output += Grid_cell.output
                Place_cell.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre, input_g=output)
                Place_cell_only.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre)
                return Place_cell.r.value, Place_cell_only.r.value

            total_time = 5000
            start_time = 1000
            indices = bm.arange(total_time)
            loc = bm.zeros(total_time) + (z_max + z_min) / 2
            input_stre = bm.zeros(total_time)
            input_stre[:start_time] = 10.

            us, us_only = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar=False)
            u = us[-1]
            u_only = us_only[-1]
            loc_num = 100
            loc_candidate = bm.linspace(z_min, z_max, loc_num, endpoint=False)

            def compute_bump_score(u, loc_candidate, map_num, Place_cell):
                def body(map_index):
                    u_place = u

                    def score_func(loc):
                        bump = Place_cell.get_bump(map_index, loc)
                        return bm.sum(bump * (u_place / bm.sum(u_place)))

                    score_candidate = jax.vmap(score_func)(loc_candidate)
                    return bm.max(score_candidate)

                bump_score = bm.for_loop(body, (bm.arange(map_num)), progress_bar=False)
                return bump_score / 0.7

            bump_score = compute_bump_score(u, loc_candidate, map_num, Place_cell) # coupled net
            bump_score_orginal_monte[j] = bump_score[0]
            bump_score_others_monte[j] = bm.max(bump_score[1:])
            bump_score_diff_monte[j] = bump_score_orginal_monte[j] - bump_score_others_monte[j]

            bump_score_only = compute_bump_score(u_only, loc_candidate, map_num, Place_cell_only) # only place net
            bump_score_orginal_monte_only[j] = bump_score_only[0]
            bump_score_others_monte_only[j] = bm.max(bump_score_only[1:])
            bump_score_diff_monte_only[j] = bump_score_orginal_monte_only[j] - bump_score_others_monte_only[j]

        print(f'Simulation trial {i+1}/{len(map_num_all)} for experiment {exp_i+1}/{exp_num}')
        for _ in tqdm(bm.arange(simulaiton_num), desc="Simulation Progress"):
            update_func(_)
            
        bump_score_diff = bm.mean(bump_score_diff_monte)
        bump_score_diff_only = bm.mean(bump_score_diff_monte_only)
        if bump_score_diff < 0.58:
            max_map_num_coupled[exp_i] = map_num
        if bump_score_diff_only < 0.58:
            max_map_num_only[exp_i] = map_num
        if bump_score_diff < 0.58 & bump_score_diff_only < 0.58:
            break

# Save the data
np.savez('stored_map.npz',
         max_map_num_coupled=max_map_num_coupled,
         max_map_num_only=max_map_num_only)

# 记录结束时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total time taken: {elapsed_time:.2f} seconds')

plt.figure()
plt.plot(place_num+grid_num*module_num, max_map_num_coupled)
plt.plot(place_num+grid_num*module_num, max_map_num_only)
plt.xlabel('neuron number')
plt.ylabel('map number')
plt.savefig('Map_number.png')