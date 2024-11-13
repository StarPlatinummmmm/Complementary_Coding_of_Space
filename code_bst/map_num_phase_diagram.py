import time
import brainstate as bst
import jax
import jax.numpy as jnp
from Network_model_bst import Place_net, Grid_net
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# Assume 'devices' is a list of available GPUs
devices = jax.devices()
print(devices)

# 记录开始时间
start_time = time.time()

# Define constants
module_num = 7
exp_num = 7
a_p = 0.5
Spacing = jnp.linspace(6, 20, module_num)
simulaiton_num = 100

def loop_func(key, grid_input_stre, z_max, map_num):
  bst.random.DEFAULT.value = key
  place_num = 600
  grid_num = 15
  Place_cell = Place_net(z_min=0, z_max=z_max, map_num=map_num, neuron_num=place_num, place_num=place_num,
                         noise_stre=0.5)
  bst.init_states(Place_cell)

  maps = Place_cell.map
  place_index = Place_cell.place_index

  Gird_module_list = [Grid_net(L=Spacing[module], maps=maps, place_index=place_index, neuron_num=grid_num, J0=5,
                               a_g=a_p / Spacing[module] * 2 * jnp.pi) for module in range(module_num)]
  for Grid_cell in Gird_module_list:
    bst.init_states(Grid_cell)

  map_index = 0
  place_only_num = place_num + grid_num * module_num
  Place_cell_only = Place_net(z_min=0, z_max=z_max, map_num=map_num, neuron_num=place_only_num,
                              place_num=place_only_num, noise_stre=0.5)
  bst.init_states(Place_cell_only)

  def run_net(indices, loc, input_stre):
    r_hpc = Place_cell.r
    output = jnp.zeros(place_num)
    with bst.environ.context(i=indices, t=indices * bst.environ.get_dt()):
      for grid_cell in Gird_module_list:
        grid_cell.update(r_hpc=r_hpc, loc=loc, input_stre=input_stre, map_index=map_index)
        output += grid_cell.output.value
      Place_cell.update(loc=loc, map_index=map_index, input_stre=input_stre, input_g=output * grid_input_stre)
      Place_cell_only.update(loc=loc, map_index=map_index, input_stre=input_stre)
    return Place_cell.r.value, Place_cell_only.r.value

  total_time = 5000
  start_time = 1000
  indices = jnp.arange(total_time)
  loc = jnp.zeros(total_time) + z_max / 2
  input_stre = jnp.zeros(total_time)
  # input_stre[:start_time] = 10.
  input_stre = input_stre.at[:start_time].set(10.)

  us, us_only = bst.transform.for_loop(run_net, indices, loc, input_stre)
  u = us[-1]
  u_only = us_only[-1]
  loc_num = 100
  loc_candidate = jnp.linspace(0, z_max, loc_num, endpoint=False)

  def compute_bump_score(u, loc_candidate, map_num, Place_cell):
    def body(map_index):
      u_place = u

      def score_func(loc):
        bump = Place_cell.get_bump(map_index, loc)
        return jnp.sum(bump * u_place)/ jnp.sqrt(jnp.sum(u_place**2)*jnp.sum(bump**2))

      score_candidate = jax.vmap(score_func)(loc_candidate)
      return jnp.max(score_candidate)

    bump_score = bst.transform.for_loop(body, (jnp.arange(map_num)))
    return bump_score / 0.7

  bump_score = compute_bump_score(u, loc_candidate, map_num, Place_cell)
  bump_score_diff_monte = bump_score[0] - jnp.max(bump_score[1:])

  bump_score_only = compute_bump_score(u_only, loc_candidate, map_num, Place_cell_only)
  bump_score_diff_monte_only = bump_score_only[0] - jnp.max(bump_score_only[1:])

  return bump_score_diff_monte, bump_score_diff_monte_only



@bst.transform.jit(static_argnums=0)
def get_score(map_num, grid_input_stre, z_max):
    key = bst.random.split_key()
    fun = partial(loop_func, map_num=map_num)
    bump_score_diff_monte, bump_score_diff_monte_only = jax.vmap(
        jax.jit(fun), in_axes=(0, None, None))(
    bst.random.split_keys(simulaiton_num), grid_input_stre, z_max
    )
    mean_coupled = jnp.mean(bump_score_diff_monte)
    mean_place = jnp.mean(bump_score_diff_monte_only)
    bst.random.DEFAULT.value = key
    return mean_coupled, mean_place  # [n_sim]
    


def get_capacity(grid_input_stre, z_max):
    bool_couple = 0
    bool_place = 0
    bool_sum = 0
    thres = 0.4
    map_num = 5
    while bool_sum < 2:
        a, c = get_score(int(map_num), float(grid_input_stre), float(z_max))
        if (c < thres) & (bool_place == 0):
            cap_place = int(map_num)
            bool_place += 1
        if (a < thres) & (bool_couple == 0):
            cap_coupled = int(map_num)
            bool_couple += 1
        bool_sum = bool_place + bool_couple
        # print(map_num)
        map_num += 1
    score_diff = a - c
    print('Capacity of coupled net:', cap_coupled)
    print('Capacity of place net:', cap_place)
    print('Score difference:', score_diff)
    return cap_coupled, cap_place, score_diff


num_grid_input = 10
num_zmax = 10
grid_input = jnp.linspace(0, 100, num_grid_input)
z_max = jnp.linspace(10, 200, num_grid_input)
# z_max = jnp.zeros(num_grid_input) + 20
cap_coupled = jnp.zeros(num_zmax, num_grid_input)
cap_place = jnp.zeros(num_zmax, num_grid_input)
score_diff = jnp.zeros(num_zmax, num_grid_input)
for i in range(num_zmax):
  for j in range(num_grid_input):
    bst.random.seed(j)
    a, b, c = get_capacity(grid_input_stre=grid_input[j], z_max=z_max[i])
    cap_coupled = cap_coupled.at[i, j].set(a)
    cap_place = cap_place.at[i, j].set(b)
    score_diff = cap_place.at[i, j].set(c)
    print('Progress:', (j+1)/num_grid_input*100,'%') # 进度显示，改一下

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total time taken: {elapsed_time:.2f} seconds')

# 画图，二维热图，横轴是z_max,纵轴是grid_input

# plt.figure()
# plt.plot(grid_input, score_diff)
# # plt.plot(grid_input, cap_place, label = 'place net')
# # plt.legend()
# plt.savefig('figures/capacity_grid_input.png')

# jnp.savez('data/map_number_grid_input.npz', cap_coupled = cap_coupled, cap_place=cap_place)




