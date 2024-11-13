import random

import brainstate as bst
import jax.numpy as jnp

bst.environ.set(dt=0.1, platform='gpu')  # length of time step


class Place_net(bst.Module):
  def __init__(self, z_min, z_max, map_num=2, neuron_num=1280, place_num=128, k=1.,
               tau=10., a_p=0.5, J0=5., noise_stre=0.05):
    super().__init__()

    # parameters
    self.tau = tau
    self.k = k  # Global inhibition
    self.a = a_p  # Range of excitatory connections
    self.J0 = J0  # maximum connection value
    self.neuron_num = neuron_num
    self.place_num = place_num
    self.map_num = map_num
    self.noise_stre = noise_stre
    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.rho = place_num / self.z_range  # The neural density
    self.dx = self.z_range / place_num  # The stimulus density
    # Construct place cell maps
    self.map, self.place_index = self.generate_maps()
    # Sample place_num neurons from all neurons and map them to feature space
    # Connections
    self.conn_mat = jnp.zeros([neuron_num, neuron_num])
    for i in range(self.map_num):
      conn_mat = self.make_conn(self.map[i])
      if i >= 1:
        mean_conn = jnp.mean(conn_mat)
        conn_mat = conn_mat - mean_conn
      self.conn_mat = self.conn_mat.at[jnp.ix_(self.place_index[i], self.place_index[i])].add(conn_mat)

  def init_state(self, batch_size=None):
    # variables
    self.r = bst.State(bst.init.param(jnp.zeros, (self.neuron_num,), batch_size))
    self.u = bst.State(bst.init.param(jnp.zeros, (self.neuron_num,), batch_size))
    self.input = bst.State(bst.init.param(jnp.zeros, (self.neuron_num,), batch_size))
    self.center = bst.State(bst.init.param(jnp.zeros, (), batch_size))
    self.center_input = bst.State(bst.init.param(jnp.zeros, (), batch_size))

  def reset_state(self, batch_size=None):
    self.r.value = bst.init.param(jnp.zeros, (self.neuron_num,), batch_size)
    self.u.value = bst.init.param(jnp.zeros, (self.neuron_num,), batch_size)
    self.input.value = bst.init.param(jnp.zeros, (self.neuron_num,), batch_size)

  def get_input(self, map_index, loc, input_stre):
    d = self.period_bound(loc - self.map[map_index, :]).reshape(-1, )
    input = input_stre * jnp.exp(-0.25 * jnp.square(d / self.a))
    return input

  def get_bump(self, map_index, center):
    d = self.period_bound(center - self.map[map_index, :]).reshape(-1, )
    bump = jnp.exp(-0.5 * jnp.square(d / self.a))
    return bump

  def generate_maps(self):
    map = jnp.zeros([self.map_num, self.place_num])
    place_index = jnp.zeros([self.map_num, self.place_num], dtype=int)
    x = jnp.linspace(self.z_min, self.z_max, self.place_num, endpoint=False)
    for i in range(self.map_num):
      map = map.at[i].set(bst.random.permutation(x))
      # random.sample(range(self.neuron_num), self.place_num)
      # sampled_integers = bst.random.randint(0, self.neuron_num, self.place_num)
      sampled_integers = bst.random.choice(self.neuron_num, self.place_num, replace=False)
      place_index = place_index.at[i].set(jnp.array(sampled_integers, dtype=int))
      place_index = place_index.at[i].set(jnp.sort(place_index[i]))
    return map, place_index

  def period_bound(self, A):
    B = jnp.where(A > self.z_range / 2, A - self.z_range, A)
    B = jnp.where(B < -self.z_range / 2, B + self.z_range, B)
    return B

  def make_conn(self, map):
    d = self.period_bound(jnp.abs(map[:, jnp.newaxis] - map))
    Jxx = self.J0 / (self.a * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * jnp.square(d / self.a))
    return Jxx

  def get_center(self, map_index):
    r0 = self.r.value[self.place_index[map_index]]
    r = jnp.where(r0 > jnp.max(r0) / 10, r0, 0)
    x = ((self.map[map_index] - self.z_min) / self.z_range - 1 / 2) * 2 * jnp.pi
    exppos = jnp.exp(1j * x)
    self.center.value = (jnp.angle(jnp.sum(exppos * r)) / 2 / jnp.pi + 1 / 2) * self.z_range + self.z_min

  def update(self, loc=0., map_index=0, input_stre=1, input_g=0.):
    # Calculate self position
    self.get_center(map_index=map_index)
    # Calculate recurrent input
    Irec = jnp.matmul(self.conn_mat, self.r.value)
    # Calculate total input
    input = self.get_input(map_index, loc, input_stre)
    self.input.value = jnp.zeros([self.neuron_num])
    # self.input.value[self.place_index[map_index]] = input
    self.input.value = self.input.value.at[self.place_index[map_index]].set(input)
    # Update neural state

    # 用新的subkey生成标准正态分布随机数组
    noise = self.noise_stre * bst.random.normal(size=(self.neuron_num,))

    du = (-self.u.value + self.input.value + Irec + noise + input_g) / self.tau * bst.environ.get_dt()
    u = self.u.value + du
    self.u.value = jnp.where(u > 0, u, 0)
    r1 = jnp.square(self.u.value)
    r2 = 1.0 + self.k * jnp.sum(r1)
    self.r.value = r1 / r2


class Grid_net(bst.Module):
  def __init__(self, L, maps, place_index, neuron_num=100, k_mec=1.,
               tau=1., a_g=1., J0=50., W0=0.1):
    super().__init__()

    # parameters
    self.tau = tau
    self.k = k_mec  # Global inhibition
    self.L = L  # Spatial Scale
    self.a = a_g  # Range of excitatory connections
    self.J0 = J0  # maximum connection value
    self.W0 = W0
    self.neuron_num = neuron_num
    self.neuron_num_hpc = maps.shape[1]
    # feature space of hpc
    self.x_hpc = maps
    self.z_range = jnp.max(maps) - jnp.min(maps)
    self.place_index = place_index
    self.map_num = maps.shape[0]
    # feature space of mec
    self.x = jnp.linspace(-jnp.pi, jnp.pi, neuron_num, endpoint=False)  # The encoded feature values
    self.rho = neuron_num / jnp.pi / 2  # The neural density
    self.dx = jnp.pi * 2 / neuron_num  # The stimulus density

    self.phase = bst.random.random((self.map_num,)) * self.L
    # Connections
    conn_mat = self.make_conn()
    self.conn_fft = jnp.fft.fft(conn_mat)  # Grid cell recurrent conn
    conn_out = jnp.zeros([self.neuron_num_hpc, self.neuron_num])
    for map_index in range(self.map_num):
      out = self.make_conn_out(map_index)
      conn_out = conn_out.at[self.place_index[map_index], :].add(out)
    self.conn_out = conn_out

  def init_state(self, batch_size=None):
    # variables
    self.r = bst.State(bst.init.param(jnp.zeros, (self.neuron_num,), batch_size))
    self.u = bst.State(bst.init.param(jnp.zeros, (self.neuron_num,), batch_size))
    self.input = bst.State(bst.init.param(jnp.zeros, (self.neuron_num,), batch_size))
    self.output = bst.State(bst.init.param(jnp.zeros, (self.neuron_num_hpc,), batch_size))
    self.center = bst.State(bst.init.param(jnp.zeros, (), batch_size))
    self.center_input = bst.State(bst.init.param(jnp.zeros, (), batch_size))

  def reset_state(self, batch_size=None):
    self.r.value = bst.init.param(jnp.zeros, (self.neuron_num,), batch_size)
    self.u.value = bst.init.param(jnp.zeros, (self.neuron_num,), batch_size)
    self.input.value = bst.init.param(jnp.zeros, (self.neuron_num,), batch_size)
    self.output.value = bst.init.param(jnp.zeros, (self.neuron_num_hpc,), batch_size)

  def xtopi(self, x, map_index):
    # x_map = x+map_index*self.L/self.map_num + self.phase
    x_map = x + self.phase[map_index]
    return (x_map % self.L) / self.L * 2 * jnp.pi - jnp.pi

  def period_bound(self, A):
    B = jnp.where(A > jnp.pi, A - 2 * jnp.pi, A)
    B = jnp.where(B < -jnp.pi, B + 2 * jnp.pi, B)
    return B

  def make_conn_out(self, map_index):
    map = self.x_hpc[map_index]
    theta_hpc = self.xtopi(map, map_index)
    D = theta_hpc[:, None] - self.x
    Dis_circ = self.period_bound(D)
    conn_out = self.W0 / (self.a * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * jnp.square(Dis_circ / self.a))
    conn_out = conn_out - jnp.mean(conn_out)
    return conn_out

  def make_conn(self):
    d = self.period_bound(jnp.abs(self.x[0] - self.x))
    Jxx = self.J0 / (self.a * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * jnp.square(d / self.a))
    return Jxx

  def get_input_pos(self, pos, map_index, input_strength):
    phase_input = self.xtopi(pos, map_index)
    d = self.period_bound(jnp.abs(self.x - phase_input))
    input = input_strength * jnp.exp(-0.25 * jnp.square(d / self.a))
    return input

  def get_center(self):
    exppos = jnp.exp(1j * self.x)
    self.center.value = jnp.angle(jnp.sum(exppos * self.r.value))
    self.center_input.value = jnp.angle(jnp.sum(exppos * self.input.value))

  def update(self, r_hpc, loc=10, input_stre=1., map_index=0):
    I_per = 0  # -50
    Ig = self.get_input_pos(pos=loc, map_index=map_index, input_strength=input_stre)
    # Calculate self position
    self.get_center()
    # Calculate hpc input
    conn_in = jnp.transpose(self.conn_out)
    input_hpc = jnp.matmul(conn_in, r_hpc.value)
    # Calculate recurrent input
    r_fft = jnp.fft.fft(self.r.value)
    Irec = jnp.real(jnp.fft.ifft(r_fft * self.conn_fft))  # real and abs
    # Calculate total input
    self.input.value = input_hpc + Ig
    # Update neural state
    du = (-self.u.value + self.input.value + Irec + I_per) / self.tau * bst.environ.get_dt()
    u = self.u.value + du
    self.u.value = jnp.where(u > 0, u, 0)
    r1 = jnp.square(self.u.value)
    r2 = 1.0 + self.k * jnp.sum(r1)
    self.r.value = r1 / r2
    self.output.value = jnp.matmul(self.conn_out, self.r.value)


class Coupled_Net(bst.Module):
  def __init__(self, HPC_model, MEC_model_list, num_module):
    super().__init__()

    self.HPC_model = HPC_model
    self.MEC_model_list = MEC_model_list
    self.num_module = num_module
    self.x_hpc = HPC_model.x
    self.a_p = HPC_model.a
    self.neuron_num_hpc = HPC_model.neuron_num
    self.I_mec = jnp.zeros([self.neuron_num_hpc, ])
    self.neuron_num_hpc = HPC_model.neuron_num
    self.W_G = jnp.ones([self.num_module, ])

  def init_state(self, batch_size=None):
    self.phase = bst.State(bst.init.param(jnp.zeros, [self.num_module], batch_size))
    self.energy = bst.State(bst.init.param(jnp.zeros, (), batch_size))

  def initial(self, Ip, Ig):
    r_hpc = jnp.zeros(self.HPC_model.neuron_num)
    I_mec = jnp.zeros(self.HPC_model.neuron_num)
    i = 0
    for MEC_model in self.MEC_model_list:
      MEC_model.update(Ig[i], r_hpc)
      i += 1
    # Update Hippocampus states
    self.HPC_model.update(Ip, I_grid=I_mec)

  def update(self, Ip, Ig):
    # Update MEC states
    r_hpc = self.HPC_model.r.value
    I_mec_module = jnp.zeros([self.neuron_num_hpc, self.neuron_num_module])
    phase = jnp.zeros([self.num_module])
    for i, MEC_model in enumerate(self.MEC_model_list):
      I_basis = jnp.matmul(MEC_model.conn_out, MEC_model.r.value)
      I_mec_module = I_mec_module.at[:, i].set(I_basis)
      MEC_model.update(Ig[i], r_hpc)
      phase = phase.at[i].set(MEC_model.center.value)
    self.phase.value = phase
    I_mec = jnp.matmul(I_mec_module, jnp.ones([self.num_module]))
    # Update Hippocampus states
    self.HPC_model.update(Ip, I_grid=I_mec)
