import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
from Network_Multiple_Maps import Place_net  # type: ignore
import jax
bm.clear_buffer_memory()
devices = jax.devices()
selected_device = devices[0]  # Indexing starts at 0, so 1 selects the second GPU

with jax.default_device(selected_device):
    # def Map_dependent_score():
    z_min = 0
    z_max = 20
    place_num = 500
    neuron_num = int(place_num)
    mapnum_all = 9
    bump_score_mean = np.zeros(mapnum_all)
    bump_score_std = np.zeros(mapnum_all)
    map_num_all = np.arange(mapnum_all)+2
    for map_i in range(mapnum_all):
        map_num = map_num_all[map_i]
        monte_num = 10
        bump_score = np.zeros([monte_num, map_num])
        max_score_pos = np.zeros([monte_num, map_num])
        for monte in range(monte_num):
            Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=neuron_num, place_num=place_num)
            maps = bm.as_numpy(Place_cell.map)
            place_index = bm.as_numpy(Place_cell.place_index)
            conn_mat = bm.as_numpy(Place_cell.conn_mat)

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
                    score_candidate = np.zeros(loc_num,)
                    for j in range(loc_num):
                        bump = Place_cell.get_bump(map_index, loc_candidate[j])
                        score_candidate[j] = bm.sum(bump*u_place)
                    bump_score[map_index] = np.max(score_candidate)
                    max_score_pos[map_index] = loc_candidate[np.argmax(score_candidate)]
                return bump_score, max_score_pos

            Place_cell.reset_state()
            total_time = 10000
            start_time = 1000
            indices = bm.arange(total_time)
            loc = bm.zeros(total_time) + (z_max + z_min) / 2
            input_stre = bm.zeros(total_time)
            input_stre[:start_time] = 10.
            u, input = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar=False)

            bump_score[monte], max_score_pos[monte] = get_bump_score(Place_cell)
        print(map_i)

        bump_score_mean_plot = np.mean(bump_score, axis=0)
        bump_score_mean[map_i] = bump_score_mean_plot[0]
        bump_score_std[map_i] = np.std(bump_score[:,0], axis=0)

        plt.figure()
        plt.bar(np.arange(map_num), bump_score_mean_plot)
        figure_name = 'figures_score/bump_score_map_num_'+str(map_num)+'.png'
        plt.savefig(figure_name)

    np.save('bump_score_mean.npy', bump_score_mean)
    np.save('bump_score_std.npy', bump_score_std)

    # 绘制误差条形图
    plt.figure()
    plt.plot(map_num_all,bump_score_mean)
    plt.errorbar(map_num_all, bump_score_mean, yerr=bump_score_std, fmt='o', ecolor='r', capsize=5, label='Data with error bars')

    # 设置图形标题和标签
    plt.title('Error Bar Plot')
    plt.xlabel('Embedded Map number')
    plt.ylabel('Bump Score')
    plt.legend()
    figure_name = 'figures_score/mapnum_dependent_score.png'
    plt.savefig(figure_name)