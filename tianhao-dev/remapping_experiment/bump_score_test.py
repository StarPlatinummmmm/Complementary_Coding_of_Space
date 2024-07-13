import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from Network_Multiple_Maps import Place_net

z_min = 0
z_max = 20
place_num = 940
neuron_num = int(place_num)
simulaiton_num = 10

map_num = 11
bump_score_orginal_monte = np.zeros(simulaiton_num)
bump_score_others_monte = np.zeros(simulaiton_num)
bump_score_diff_monte = np.zeros(simulaiton_num)
for j in range(simulaiton_num):
    Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=neuron_num, place_num=place_num, noise_stre=0.5)
    maps = bm.as_numpy(Place_cell.map) 
    # How place cells are mapped to the feature space: map_num * place_num
    place_index = bm.as_numpy(Place_cell.place_index) 
    # which sub-group of neurons are seleted as place cells: map_num * place_num
    conn_mat = bm.as_numpy(Place_cell.conn_mat) # connection matrix neuron_num * neuron_num

    map_index = 0
    def run_net(indices, loc, input_stre):
        Place_cell.step_run(indices, loc=loc, map_index = map_index, input_stre=input_stre)
        u = Place_cell.u
        input = Place_cell.input
        return u, input

    total_time = 10000
    start_time = 1000
    indices = bm.arange(total_time)
    loc = bm.zeros(total_time) + (z_max+z_min)/2
    input_stre = bm.zeros(total_time) 
    input_stre[:start_time] = 10.
    u, input = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar = False)

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
    print('simulation trail:', j/simulaiton_num)

    print('selected bump score:', bump_score_orginal_monte[j])
    print('other bump score:', bump_score_others_monte[j])
    print('bump score diff:', bump_score_diff_monte[j])

print('mean bump score:', np.mean(bump_score_orginal_monte))
print('other bump score:', np.mean(bump_score_others_monte))
print('bump score diff:', np.mean(bump_score_diff_monte))

