import time
import brainstate as bst
import jax
import jax.numpy as jnp
from Network_model_bst import Place_net, Grid_net
from functools import partial
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
num_map = 20
simulaiton_num = 100
place_num = jnp.linspace(400, 800, exp_num, dtype=int)
grid_num = place_num // 40

def loop_func(key,grid_input_stre,z_max,map_num):
  bst.random.DEFAULT.value = key
  place_num = 600
  grid_num = 15
#   z_max = 20
#   grid_input_stre = 1.
#   map_num = param
#   map_num = 10
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
    return bump_score/0.7

  bump_score = compute_bump_score(u, loc_candidate, map_num, Place_cell)
  bump_score_diff_monte = bump_score[0] - jnp.max(bump_score[1:])

  bump_score_only = compute_bump_score(u_only, loc_candidate, map_num, Place_cell_only)
  bump_score_diff_monte_only = bump_score_only[0] - jnp.max(bump_score_only[1:])

  return bump_score_diff_monte, bump_score_diff_monte_only


num_grid_input = 30
grid_input = jnp.linspace(0, 50, num_grid_input)
z_max = jnp.zeros(num_grid_input) + 20
bump_score_diff = jnp.zeros(num_grid_input)



bst.random.seed(0)

@bst.transform.jit(static_argnums=0)
def get_score(map_num, grid_input_stre, z_max):
    key = bst.random.split_key()
    fun = partial(loop_func, map_num=map_num)
    bump_score_diff_monte, bump_score_diff_monte_only = jax.vmap(
        jax.jit(fun), in_axes=(0, None, None))(
    bst.random.split_keys(simulaiton_num), grid_input_stre, z_max
    )
    mean_coupled = jnp.mean(bump_score_diff_monte)
    std_coupled = jnp.std(bump_score_diff_monte)
    mean_place = jnp.mean(bump_score_diff_monte_only)
    std_place = jnp.std(bump_score_diff_monte_only)
    bst.random.DEFAULT.value = key
    return mean_coupled, std_coupled, mean_place, std_place  # [n_sim]
    



map_num_all = []
mean_coupled = []
std_coupled = []
mean_place = []
std_place = []
bool_variable = 0
thres = 0.4
map_num = 5
for i in range(20):
    map_num = i+2
    a, b, c, d = get_score(int(map_num), float(40), float(20))
    mean_coupled.append(a)
    std_coupled.append(b)
    mean_place.append(c)
    std_place.append(d)
    map_num_all.append(int(map_num))
    if (c < thres) & (bool_variable == 0):
        cap_place = int(map_num)
        bool_variable += 1
    if (a < thres) & (bool_variable == 1):
        cap_coupled = int(map_num)
        bool_variable += 1
    print((i+1)/20)

print('Capacity of coupled net:', cap_coupled)
print('Capacity of place net:', cap_place)
mean_coupled = jnp.array(mean_coupled)
std_coupled = jnp.array(std_coupled)
mean_place = jnp.array(mean_place)
std_place = jnp.array(std_place)
map_num_all = jnp.array(map_num_all)
jnp.savez('data/scores.npz', mean_coupled = mean_coupled, std_coupled=std_coupled,
          mean_place = mean_place, std_place=std_place)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total time taken: {elapsed_time:.2f} seconds')


import matplotlib.pyplot as plt
def shaded_errorbar(ax, x, y, yerr, label=None, color=None, alpha_fill=0.3):
    ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha_fill)

plt.rcParams.update({'font.size': 15, 'font.family': 'Arial'})
plt.figure(figsize=(4, 4))
shaded_errorbar(plt.gca(), map_num_all, mean_coupled, std_coupled, label='800 Place cells with 20*7 grid cells', color='g')
shaded_errorbar(plt.gca(), map_num_all, mean_place, std_place, label='940 Place cells', color='b')

plt.xlabel('Embedded Map Number')
plt.ylabel('Bump Score Differences')
plt.title('Bump Score Differences')
plt.xticks(map_num_all[::2])
# plt.legend()
plt.grid(True)
# Save the plot
plt.tight_layout()
plt.savefig('figures/Bump_score_difference.png')

