import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from Network_Multiple_Maps import Place_net
from matplotlib.animation import FuncAnimation


z_min = -np.pi
z_max = np.pi 
place_num = 1280
neuron_num = int(5 * place_num)
map_num = 2

Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=neuron_num, place_num=place_num)
maps = bm.as_numpy(Place_cell.map) # How place cells are mapped to the feature space: map_num * place_num
place_index = bm.as_numpy(Place_cell.place_index) # which sub-group of neurons are seleted as place cells: map_num * place_num
conn_mat = bm.as_numpy(Place_cell.conn_mat) # connection matrix neuron_num * neuron_num

map_index = 0
def run_net(indices, loc, input_stre):
    Place_cell.step_run(indices, loc=loc, map_index = map_index, input_stre=input_stre)
    u = Place_cell.u
    input = Place_cell.input
    return u, input

total_time = 5000
start_time = 1000
indices = bm.arange(total_time)
loc = bm.zeros(total_time)
input_stre = bm.zeros(total_time)
input_stre[:start_time] = 1.
u, input = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar = True)



_, ax = plt.subplots(1,map_num,figsize=(map_num*4, 4))
for map_index in range(map_num):
    u_place = bm.as_numpy(u[:, place_index[map_index]])
    sorted_indices = np.argsort(maps[map_index])
    sorted_maps = maps[map_index, sorted_indices]
    sorted_u_place = u_place[:, sorted_indices]
    # ax.plot(sorted_maps, sorted_u_place[-1,:])
    ax[map_index].plot(sorted_maps, sorted_u_place[-1,:])
    # ax[map_index].set_ylim([0, 5])
    ax[map_index].set_title('Neural activity (map'+str(map_index)+')')

n_step = 10
map_index = 0
u_place = bm.as_numpy(u[:, place_index[map_index]])
sorted_indices = np.argsort(maps[map_index, :])
sorted_maps = maps[map_index, sorted_indices]
sorted_u_place = u_place[:, sorted_indices]

fig, ax_ani = plt.subplots(1, 1, figsize=(5, 3), dpi=100)

data = sorted_u_place[::n_step, :]
max_data = np.max(data)*1.5
T = data.shape[0]

line1, = ax_ani.plot([], [])
ax_ani.set_title("Bump activity")
ax_ani.set_xlim([-np.pi, np.pi])
ax_ani.set_ylim([-1, max_data])

def update(frame):
    y1 = data[frame].flatten()
    line1.set_data(sorted_maps, y1)
    return line1

ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)





plt.show()
# ani.save(aniname, writer="Pillow", fps=10)