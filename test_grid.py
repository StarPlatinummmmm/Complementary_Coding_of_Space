import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from Network_Multiple_Maps import Place_net, Grid_net
z_min = 0
z_max = 20
place_num = 800
grid_num = 30
map_num = 2

Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=place_num, place_num=place_num, noise_stre=0.5)
maps = bm.as_numpy(Place_cell.map) 
place_index = bm.as_numpy(Place_cell.place_index) 
Grid_cell = Grid_net(L = 5, maps=maps, place_index=place_index, neuron_num=grid_num, J0=5)
conn_out = Grid_cell.conn_out
