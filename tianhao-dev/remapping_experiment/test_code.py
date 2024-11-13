import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from Network_model_bst import Place_net, Grid_net
import time
import gc  # Import garbage collection module
from jax import vmap
import brainstate as bst

# Assume 'devices' is a list of available GPUs
devices = jax.devices()
print(devices)

# Select a specific device, for example, the second GPU
selected_device = devices[4] 

# 记录开始时间
start_time = time.time()

# Define constants
module_num = 7
exp_num = 7
a_p = 0.5
Spacing = jnp.linspace(6, 20, module_num)
num_map = 20
map_num_all = jnp.arange(num_map) + 2  # 2-17
simulaiton_num = 50
place_num = jnp.linspace(400, 800, exp_num, dtype=int)
grid_num = place_num // 40

map_num = 10
place_num = 800
grid_num = 20
z_max = 20
grid_input_stre=1.

def loop_func(j):
    map_num = 10
    place_num = 800
    grid_num = 20
    z_max = 20
    grid_input_stre=1.
    Place_cell = Place_net(z_min=0, z_max=z_max, map_num=map_num, neuron_num=place_num, place_num=place_num, noise_stre=0.5)
    maps = Place_cell.map
    place_index = Place_cell.place_index

    Gird_module_list = [Grid_net(L=Spacing[module], maps=maps, place_index=place_index, neuron_num=grid_num, J0=5,
                                a_g=a_p / Spacing[module] * 2 * jnp.pi) for module in range(module_num)]
    map_index = 0
    place_only_num = place_num + grid_num*module_num
    Place_cell_only = Place_net(z_min=0, z_max=z_max, map_num=map_num, neuron_num=place_only_num, place_num=place_only_num, noise_stre=0.5) 

    def run_net(indices, loc, input_stre):
        r_hpc = Place_cell.r
        output = jnp.zeros(place_num)
        for Grid_cell in Gird_module_list:
            Grid_cell.step_run(indices, r_hpc=r_hpc, loc=loc, input_stre=input_stre, map_index=map_index)
            output += Grid_cell.output
        Place_cell.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre, input_g=output*grid_input_stre)

        Place_cell_only.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre)
        return Place_cell.r.value, Place_cell_only.r.value

    total_time = 5000
    start_time = 1000
    indices = jnp.arange(total_time)
    loc = jnp.zeros(total_time) + z_max / 2
    input_stre = jnp.zeros(total_time)
    input_stre[:start_time] = 10.

    us, us_only = jnp.for_loop(run_net, (indices, loc, input_stre), progress_bar=False)
    u = us[-1]
    u_only = us_only[-1]
    loc_num = 100
    loc_candidate = jnp.linspace(0, z_max, loc_num, endpoint=False)

    def compute_bump_score(u, loc_candidate, map_num, Place_cell):
        def body(map_index):
            u_place = u

            def score_func(loc):
                bump = Place_cell.get_bump(map_index, loc)
                return jnp.sum(bump * (u_place / jnp.sum(u_place)))

            score_candidate = jax.vmap(score_func)(loc_candidate)
            return jnp.max(score_candidate)

        bump_score = jnp.for_loop(body, (jnp.arange(map_num)), progress_bar=False)
        return bump_score / 0.7

    bump_score = compute_bump_score(u, loc_candidate, map_num, Place_cell)
    bump_score_diff_monte = bump_score[0] - jnp.max(bump_score[1:])

    bump_score_only = compute_bump_score(u_only, loc_candidate, map_num, Place_cell_only)
    bump_score_diff_monte_only = bump_score_only[0] - jnp.max(bump_score_only[1:])

    # Release memory by deleting objects and calling garbage collector
    del Place_cell
    del Gird_module_list
    del Place_cell_only
    gc.collect()
    return bump_score_diff_monte, bump_score_diff_monte_only


num_grid_input = 30
grid_input = jnp.linspace(0,50,num_grid_input)
z_max = jnp.zeros(num_grid_input)+20
bump_score_diff = jnp.zeros((num_grid_input))

monte = jnp.arange(simulaiton_num)
bump_score_diff_monte, bump_score_diff_monte_only = bst.transform.map(loop_func, monte)

# bump_score_diff_monte = np.zeros(simulaiton_num)
# bump_score_diff_monte_only = np.zeros(simulaiton_num)
# for j in range(simulaiton_num):
#     bump_score_diff_monte[j], bump_score_diff_monte_only[j] = loop_func(j)
#     print(j/simulaiton_num)
# np.savez('bump_score_forloop.npz', bump_score_diff_monte = bump_score_diff_monte, bump_score_diff_monte_only=bump_score_diff_monte_only)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total time taken: {elapsed_time:.2f} seconds')
print('mean diff:', jnp.mean(bump_score_diff_monte))
print('std diff:', jnp.std(bump_score_diff_monte))
print('mean diff only:', jnp.mean(bump_score_diff_monte_only))
print('std diff only:', jnp.std(bump_score_diff_monte_only))







