import brainstate as bst
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

bst.environ.set(dt=0.2, platform='cpu')  # length of time step


class Place_net(bst.Module):
  def __init__(self, z_min, z_max, num=1000, k=1., tau=10., a_p=4., J0=50.):
    super().__init__()

    # parameters
    self.tau = tau
    self.k = k  # Global inhibition
    self.a = a_p  # Range of excitatory connections
    self.J0 = J0  # maximum connection value
    self.num = num
    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = jnp.linspace(z_min, z_max, num, endpoint=False)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # Connections
    conn_mat = self.make_conn()
    self.conn_fft = jnp.fft.fft(conn_mat)  # recurrent conn

  def init_state(self, batch_size=None):
    # variables
    self.r = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.r_mec = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.u = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.v = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.input = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.center = bst.State(bst.init.param(jnp.zeros, (), batch_size))

  def reset_state(self, batch_size=None):
    self.r.value = bst.init.param(jnp.zeros, (self.num,), batch_size)
    self.u.value = bst.init.param(jnp.zeros, (self.num,), batch_size)
    self.v.value = bst.init.param(jnp.zeros, (self.num,), batch_size)
    self.input.value = bst.init.param(jnp.zeros, (self.num,), batch_size)

  def period_bound(self, A):
    B = jnp.where(A > self.z_range / 2, A - self.z_range, A)
    B = jnp.where(B < -self.z_range / 2, B + self.z_range, B)
    return B

  def make_conn(self):
    d = self.period_bound(jnp.abs(self.x[0] - self.x))
    Jxx = self.J0 / (self.a * np.sqrt(2 * np.pi)) * jnp.exp(-0.5 * jnp.square(d / self.a))
    return Jxx

  def get_center(self):
    r = self.r.value
    x = np.linspace(-np.pi, np.pi, self.num, endpoint=False)
    exppos = jnp.exp(1j * x)
    self.center.value = (jnp.angle(jnp.sum(exppos * r)) / 2 / np.pi + 1 / 2) * self.z_range + self.z_min

  def update(self, Ip, I_grid):
    # Calculate self position
    self.get_center()
    # Calculate recurrent input
    r_fft = jnp.fft.fft(self.r.value)
    Irec = jnp.real(jnp.fft.ifft(r_fft * self.conn_fft))  # real and abs
    # Calculate total input
    self.input.value = I_grid + Irec + Ip
    # Update neural state
    u = self.u.value + (-self.u.value + self.input.value) / self.tau * bst.environ.get_dt()
    self.u.value = jnp.where(u > 0, u, 0)
    r1 = jnp.square(self.u.value)
    r2 = 1.0 + self.k * jnp.sum(r1)
    self.r.value = r1 / r2


class Grid_net(bst.Module):
  def __init__(self, L, z_min, z_max, num=100, num_hpc=1000, k_mec=1., tau=1., a_g=1., J0=50., W0=0.1):
    super().__init__()

    # parameters
    self.tau = tau
    self.k = k_mec  # Global inhibition
    self.L = L  # Spatial Scale
    self.a = a_g  # Range of excitatory connections
    self.J0 = J0  # maximum connection value
    self.W0 = W0
    self.num = num
    self.num_hpc = num_hpc
    # feature space
    self.x_hpc = np.linspace(z_min, z_max, num_hpc, endpoint=False)  # The encoded feature values
    self.x = np.linspace(-np.pi, np.pi, num, endpoint=False)  # The encoded feature values
    self.rho = num / np.pi / 2  # The neural density
    self.dx = np.pi * 2 / num  # The stimulus density

    # Connections
    conn_mat = self.make_conn()
    self.conn_fft = jnp.fft.fft(conn_mat)  # Grid cell recurrent conn
    self.conn_out = self.make_conn_out()  # From grid cells to place cells

  def init_state(self, batch_size=None):
    # variables
    self.r = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.u = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.v = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.input = bst.State(bst.init.param(jnp.zeros, (self.num,), batch_size))
    self.center = bst.State(bst.init.param(jnp.zeros, (), batch_size))
    self.center_input = bst.State(bst.init.param(jnp.zeros, (), batch_size))

  def reset_state(self, batch_size=None):
    self.r.value = bst.init.param(jnp.zeros, (self.num,), batch_size)
    self.u.value = bst.init.param(jnp.zeros, (self.num,), batch_size)
    self.v.value = bst.init.param(jnp.zeros, (self.num,), batch_size)
    self.input.value = bst.init.param(jnp.zeros, (self.num,), batch_size)

  def xtopi(self, x):
    return (x % self.L) / self.L * 2 * np.pi - np.pi

  def period_bound(self, A):
    B = jnp.where(A > jnp.pi, A - 2 * jnp.pi, A)
    B = jnp.where(B < -jnp.pi, B + 2 * jnp.pi, B)
    return B

  def make_conn_out(self):
    theta_hpc = self.xtopi(self.x_hpc)
    D = theta_hpc[:, None] - self.x
    Dis_circ = self.period_bound(D)
    conn_out = self.W0 / (self.a * np.sqrt(2 * np.pi)) * jnp.exp(-0.5 * jnp.square(Dis_circ / self.a))
    return conn_out

  def make_conn(self):
    d = self.period_bound(jnp.abs(self.x[0] - self.x))
    Jxx = self.J0 / (self.a * np.sqrt(2 * np.pi)) * jnp.exp(-0.5 * jnp.square(d / self.a))
    return Jxx

  def get_center(self):
    exppos = jnp.exp(1j * self.x)
    self.center.value = jnp.angle(jnp.sum(exppos * self.r.value))
    self.center_input.value = jnp.angle(jnp.sum(exppos * self.input.value))

  def update(self, Ig, r_hpc):
    # Calculate self position
    self.get_center()
    # Calculate hpc input
    conn_in = jnp.transpose(self.conn_out)
    input_hpc = jnp.matmul(conn_in, r_hpc)
    # Calculate recurrent input
    r_fft = jnp.fft.fft(self.r.value)
    Irec = jnp.real(jnp.fft.ifft(r_fft * self.conn_fft))  # real and abs
    # Calculate total input
    self.input.value = input_hpc + Irec + Ig
    # Update neural state
    u = self.u.value + (-self.u.value + self.input.value) / self.tau * bst.environ.get_dt()
    self.u.value = jnp.where(u > 0, u, 0)
    r1 = jnp.square(self.u.value)
    r2 = 1.0 + self.k * jnp.sum(r1)
    self.r.value = r1 / r2


class Coupled_Net(bst.Module):
  def __init__(self, num_module, L, num_g, Lambda, a_g, k_g, tau_g, J_g, J_pg,
               num_p, a_p, k_p, tau_p, J_p):
    super().__init__()

    G_CANNs = bst.visible_module_list()
    for i in range(num_module):
      G_CANNs.append(Grid_net(z_min=0, z_max=L, num=num_g, num_hpc=num_p, L=Lambda[i],
                              a_g=a_g[i], k_mec=k_g[i], tau=tau_g[i], J0=J_g, W0=J_pg))

    self.HPC_model = Place_net(z_min=0, z_max=L, num=num_p, a_p=a_p, k=k_p, tau=tau_p, J0=J_p)
    self.MEC_model_list = G_CANNs
    self.num_module = num_module
    self.L = L
    self.num_g = num_g
    self.num_p = num_p
    self.a_g = a_g
    self.a_p = a_p

  def init_state(self, batch_size=None):
    self.phase = bst.State(bst.init.param(jnp.zeros, [self.num_module], batch_size))
    self.energy = bst.State(bst.init.param(jnp.zeros, (), batch_size))
    self.I_mec = bst.State(bst.init.param(jnp.zeros, [self.num_p], batch_size))

  def initial(self, alpha_p, alpha_g, Ip, Ig):
    # Update MEC states
    r_hpc = jnp.zeros(self.HPC_model.num)
    I_mec = jnp.zeros(self.HPC_model.num)
    i = 0
    for MEC_model in self.MEC_model_list:
      MEC_model.update(alpha_g * Ig[i], r_hpc)
      i += 1
    # Update Hippocampus states
    self.HPC_model.update(alpha_p * Ip, I_grid=I_mec)

  def Get_Energy(self, alpha_p, alpha_g, Ip, Ig):
    ug = self.MEC_model_list[0].u.value
    up = self.HPC_model.u.value
    rg = self.MEC_model_list[0].r.value
    rp = self.HPC_model.r.value
    # Bump height
    Ag = jnp.max(ug)
    Ap = jnp.max(up)
    Rg = jnp.max(rg)
    Rp = jnp.max(rp)
    # Parameters
    rho_g = self.MEC_model_list[0].rho
    tau_g = self.MEC_model_list[0].tau
    tau_p = self.HPC_model.tau
    rho_p = self.HPC_model.rho
    a_p = self.HPC_model.a
    # Sigma
    Lambda = jnp.zeros(self.num_module, )
    sigma_phi = jnp.zeros(self.num_module)
    for i in range(self.num_module):
      J_pg = self.MEC_model_list[i].W0
      Lambda = Lambda.at[i].set(self.MEC_model_list[i].L)
      sigma_phi = sigma_phi.at[i].set(1 / ((Lambda[i] / 2 / jnp.pi) * jnp.sqrt(J_pg * rho_g * Rg / (4 * Ap * tau_p))))
    sigma_p = jnp.sqrt(jnp.sqrt(jnp.pi) * Ap ** 3 * rho_p * tau_p / (a_p * alpha_p))
    # Feature space
    place_x = self.HPC_model.x
    theta = self.MEC_model_list[0].x

    # Network decoding
    z = self.HPC_model.center.value
    psi_z = jnp.mod(z / Lambda, 1) * 2 * jnp.pi
    phi = jnp.zeros(self.num_module, )
    for i in range(self.num_module):
      phi = phi.at[i].set(self.MEC_model_list[i].center.value)

    # 圆周距离函数
    def circ_dis(phi_1, phi_2):
      dis = phi_1 - phi_2
      dis = jnp.where(dis > jnp.pi, dis - 2 * jnp.pi, dis)
      dis = jnp.where(dis < -jnp.pi, dis + 2 * jnp.pi, dis)
      return dis

    # Calculate log posterior
    log_prior = 0
    log_likelihood_grid = 0
    for i in range(self.num_module):
      a_g = self.MEC_model_list[i].a
      sigma_g = jnp.sqrt(jnp.sqrt(jnp.pi) * Ag ** 3 * rho_g * tau_g / (a_g * alpha_g))
      dis_1 = circ_dis(theta, phi[i])
      fg = jnp.exp(-dis_1 ** 2 / (4 * a_g ** 2))
      log_likelihood_grid -= jnp.sum((Ig[i, :] - fg) ** 2) / sigma_g ** 2
      dis_2 = circ_dis(phi[i], psi_z[i])
      log_prior -= 1 / (sigma_phi[i] ** 2) * jnp.exp(-dis_2 ** 2 / 8 / a_g ** 2) * dis_2 ** 2
    dis_x = place_x - z
    fp = jnp.exp(-dis_x ** 2 / (4 * a_p ** 2))
    log_likelihood_place = -jnp.sum((Ip - fp) ** 2) / sigma_p ** 2
    log_posterior = log_likelihood_grid + log_prior + log_likelihood_place
    # Energy function
    self.energy.value = - log_posterior

  def update(self, alpha_p, alpha_g, Ip, Ig):
    self.Get_Energy(alpha_p, alpha_g, Ip, Ig)
    # Update MEC states
    r_hpc = self.HPC_model.r.value
    I_mec_module = jnp.zeros([self.HPC_model.num, self.num_module])
    phase = jnp.zeros([self.num_module])
    for i, MEC_model in enumerate(self.MEC_model_list):
      I_basis = jnp.matmul(MEC_model.conn_out, MEC_model.r.value)
      I_mec_module = I_mec_module.at[:, i].set(I_basis)
      MEC_model.update(alpha_g * Ig[i], r_hpc)
      phase = phase.at[i].set(MEC_model.center.value)
    self.phase.value = phase
    index = jnp.argmax(r_hpc)
    kernel = jnp.zeros(self.num_p)
    for j in range(10):
        kernel.at[j+index-5].set(1)
    I_mec = jnp.matmul(I_mec_module, jnp.ones([self.num_module])) * kernel
    self.I_mec.value = I_mec
    # Update Hippocampus states
    self.HPC_model.update(alpha_p * Ip, I_grid=I_mec)

  def Net_decoding(self, z_truth, phi_truth, Ip, Ig, alpha_p=0.05, alpha_g=0.05):
    def initial_net(Ip, Ig):
      self.initial(alpha_p=1, alpha_g=1, Ip=Ip, Ig=Ig)

    def run_net(i, Ip, Ig):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(alpha_p=1, alpha_g=1, Ip=Ip, Ig=Ig)
      phi_decode = self.phase.value
      z_decode = self.HPC_model.center.value
      rp = self.HPC_model.r.value
      up = self.HPC_model.u.value
      rg = jnp.zeros([self.num_module, self.num_g])
      ug = jnp.zeros([self.num_module, self.num_g])
      for mi in range(self.num_module):
        rg = rg.at[mi].set(self.MEC_model_list[mi].r.value)
        ug = ug.at[mi].set(self.MEC_model_list[mi].u.value)
      return z_decode, phi_decode, rp, up, rg, ug

    T_init = 500
    z0 = z_truth
    phi_0 = phi_truth
    fg = jnp.zeros((self.num_module, self.num_g))
    theta = np.linspace(-np.pi,np.pi,self.num_g,endpoint=False)
    for i in range(self.num_module):
      dis_theta = circ_dis(theta, phi_0[i])
      fg = fg.at[i].set(jnp.exp(-dis_theta ** 2 / (4 * self.a_g[i] ** 2)))
    x = np.linspace(0, self.L, self.num_p, endpoint=False)
    dis_x = x - z0
    fp = jnp.exp(-dis_x ** 2 / (4 * self.a_p ** 2))
    I_place = 1 * jnp.repeat(fp[np.newaxis, :], T_init, axis=0)
    I_grid = 1 * jnp.repeat(fg[np.newaxis, :, :], T_init, axis=0)
    I_place = I_place.at[int(T_init / 3):, :].set(0)
    I_grid = I_grid.at[int(T_init / 3):, :, :].set(0)

    bst.transform.for_loop(initial_net, I_place, I_grid)
    T = 2000
    indices = np.arange(T)
    I_place = alpha_p * jnp.repeat(Ip[np.newaxis, :], T, axis=0)
    I_grid = alpha_g * jnp.repeat(Ig[np.newaxis, :, :], T, axis=0)
    z_record, phi_record, rp, up, rg, ug = bst.transform.for_loop(run_net, indices, I_place, I_grid)
    return z_record, phi_record, up, rp, ug, rg


# 圆周距离函数
def circ_dis(phi_1, phi_2):
  dis = phi_1 - phi_2
  dis = jnp.where(dis > jnp.pi, dis - 2 * jnp.pi, dis)
  dis = jnp.where(dis < -jnp.pi, dis + 2 * jnp.pi, dis)
  return dis