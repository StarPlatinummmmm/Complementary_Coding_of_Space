import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
from Network_Multiple_Maps import Place_net  # type: ignore

z_min = 0
z_max = 20
place_num = 640
neuron_num = int(place_num)
map_num = 10

Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=neuron_num, place_num=place_num)
maps = bm.as_numpy(Place_cell.map)
place_index = bm.as_numpy(Place_cell.place_index)
conn_mat = bm.as_numpy(Place_cell.conn_mat)

_, ax = plt.subplots(1, map_num, figsize=(map_num * 4, 4))
for i in range(map_num):
    sorted_indices = np.argsort(maps[i])
    sorted_maps = maps[i, sorted_indices]
    sorted_place_index = place_index[i, sorted_indices]
    sorted_conn_mat = conn_mat[np.ix_(sorted_place_index, sorted_place_index)]

    ax[i].pcolormesh(sorted_maps, sorted_maps, sorted_conn_mat, cmap='viridis')
    ax[i].set_title('Conn matrix (map' + str(i) + ')')
plt.savefig('./figures_remapping/connection_matrix')

map_index = 0


def run_net(indices, loc, input_stre):
    Place_cell.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre)
    u = Place_cell.r
    input = Place_cell.input
    return u, input


def get_bump_score(Place_cell):
    u = Place_cell.r
    loc_num = 100
    loc_candidate = np.linspace(z_min,z_max,loc_num,endpoint=False)
    bump_score = np.zeros(map_num,)
    max_score_pos = np.zeros(map_num,)
    for map_index in range(map_num):
        u_place = bm.as_numpy(u[place_index[map_index]])
        sorted_indices = np.argsort(maps[map_index])
        sorted_u_place = u_place[sorted_indices]/bm.sum(u_place)
        score_candidate = np.zeros(loc_num,)
        for j in range(loc_num):
            bump = Place_cell.get_bump(map_index, loc_candidate[j])
            sorted_bump_normalize = bump[sorted_indices]
            score_candidate[j] = bm.sum(sorted_bump_normalize*sorted_u_place)
        bump_score[map_index] = np.max(score_candidate)
        max_score_pos[map_index] = loc_candidate[np.argmax(score_candidate)]
    return bump_score, max_score_pos

monte_num = 100
# bump_score = np.zeros([monte_num, map_num])
# max_score_pos = np.zeros([monte_num, map_num])
# for monte in range(monte_num):
    # Place_cell.reset_state()
total_time = 10000
start_time = 1000
indices = bm.arange(total_time)
loc = bm.zeros(total_time) + (z_max + z_min) / 2
input_stre = bm.zeros(total_time)
input_stre[:start_time] = 10.
u, input = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar=True)

bump_score, max_score_pos = get_bump_score(Place_cell)

plt.figure()
plt.bar(np.arange(map_num), bump_score)
plt.savefig('./figures_remapping/bump_score.png')

if map_num>5: 
    map_num = 5
_, ax = plt.subplots(1, map_num, figsize=(map_num * 4, 4))
for map_index in range(map_num):
    u_place = bm.as_numpy(u[:, place_index[map_index]])
    sorted_indices = np.argsort(maps[map_index])
    sorted_maps = maps[map_index, sorted_indices]
    sorted_u_place = u_place[:, sorted_indices]
    bump = Place_cell.get_bump(map_index, max_score_pos[map_index])
    sorted_bump_normalize = bump[sorted_indices]
    bump_height = bm.max(sorted_u_place[-1, :])
    sorted_bump = sorted_bump_normalize*bump_height

    ax[map_index].plot(sorted_maps, sorted_u_place[-1, :])
    ax[map_index].plot(sorted_maps, sorted_bump)
    ax[map_index].set_title('Neural activity (map' + str(map_index) + ')')
    ax[map_index].set_ylim(0, np.max(u[-1, :])+0.03)

plt.savefig('./figures_remapping/bump_activity.png')


# # 动图部分
# fig, ax = plt.subplots(1, map_num, figsize=(map_num * 4, 4))

# def animate(t):
#     for map_index in range(map_num):
#         u_place = bm.as_numpy(u[:, place_index[map_index]])
#         sorted_indices = np.argsort(maps[map_index])
#         sorted_maps = maps[map_index, sorted_indices]
#         sorted_u_place = u_place[:, sorted_indices]
#         ax[map_index].clear()
#         ax[map_index].plot(sorted_maps, sorted_u_place[t, :])
#         ax[map_index].set_title(f'Neural activity (map {map_index}) at time {t}')
#         # ax[map_index].set_ylim(0, np.max(u[-1, :])+1)
#     plt.tight_layout()

# anim = FuncAnimation(fig, animate, frames=range(0, total_time, 10), interval=200)
# anim.save('./figures_remapping/neural_activity.gif', writer='Pillow')
# # plt.show()
