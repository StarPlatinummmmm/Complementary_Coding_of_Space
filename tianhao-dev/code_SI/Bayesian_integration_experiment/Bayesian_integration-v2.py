import brainstate as bst
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

bst.environ.set(dt=0.2, platform='cpu')  # length of time step


# grid spacing
lambda_1 = 3
lambda_2 = 4
lambda_3 = 5
Lambda = np.array([lambda_1, lambda_2, lambda_3])
L = lambda_1 * lambda_2 * lambda_3
# cell number
num_p = int(200)
rho_p = num_p / L
rho_g = rho_p
num_g = int(rho_g * 2 * np.pi)  # 为了让两个网络的rho相等
M = len(Lambda)
# feature space
x = np.linspace(0, L, num_p, endpoint=False)
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
# connection range
a_p = 0.3
a_g = a_p / Lambda * 2 * np.pi
# connection strength
J_p = 20
J_g = J_p
J_pg = J_p / 50
# divisive normalization
k_p = 20.
k_g = Lambda / 2 / np.pi * k_p
# time constants
tau_p = 1
tau_g = 2 * np.pi * tau_p / Lambda
# input_strength
alpha_p = 0.05
alpha_g = 0.05
noise_ratio = 0.007
neural_noise_sigma = 0.12
trial_num = 50
z_truth = 30
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi

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
  def __init__(self, num_module):
    super().__init__()

    G_CANNs = bst.visible_module_list()
    for i in range(M):
      G_CANNs.append(Grid_net(z_min=0, z_max=L, num=num_g, num_hpc=num_p, L=Lambda[i],
                              a_g=a_g[i], k_mec=k_g[i], tau=tau_g[i], J0=J_g, W0=J_pg))

    self.HPC_model = Place_net(z_min=0, z_max=L, num=num_p, a_p=a_p, k=k_p, tau=tau_p, J0=J_p)
    self.MEC_model_list = G_CANNs
    self.num_module = num_module

  def init_state(self, batch_size=None):
    self.phase = bst.State(bst.init.param(jnp.zeros, [self.num_module], batch_size))
    self.energy = bst.State(bst.init.param(jnp.zeros, (), batch_size))

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
    I_mec = jnp.matmul(I_mec_module, jnp.ones([self.num_module]))
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
      rg = jnp.zeros([M, num_g])
      ug = jnp.zeros([M, num_g])
      for mi in range(M):
        rg = rg.at[mi].set(self.MEC_model_list[mi].r.value)
        ug = ug.at[mi].set(self.MEC_model_list[mi].u.value)
      return z_decode, phi_decode, rp, up, rg, ug

    T_init = 500
    z0 = z_truth
    phi_0 = phi_truth
    fg = jnp.zeros((M, num_g))
    for i in range(M):
      dis_theta = circ_dis(theta, phi_0[i])
      fg = fg.at[i].set(jnp.exp(-dis_theta ** 2 / (4 * a_g[i] ** 2)))
    x = np.linspace(0, L, num_p, endpoint=False)
    dis_x = x - z0
    fp = jnp.exp(-dis_x ** 2 / (4 * a_p ** 2))
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


# grid spacing
lambda_1 = 3
lambda_2 = 4
lambda_3 = 5
Lambda = np.array([lambda_1, lambda_2, lambda_3])
L = lambda_1 * lambda_2 * lambda_3
# cell number
num_p = int(200)
rho_p = num_p / L
rho_g = rho_p
num_g = int(rho_g * 2 * np.pi)  # 为了让两个网络的rho相等
M = len(Lambda)
# feature space
x = np.linspace(0, L, num_p, endpoint=False)
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
# connection range
a_p = 0.3
a_g = a_p / Lambda * 2 * np.pi
# connection strength
J_p = 20
J_g = J_p
J_pg = J_p / 50
# divisive normalization
k_p = 20.
k_g = Lambda / 2 / np.pi * k_p
# time constants
tau_p = 1
tau_g = 2 * np.pi * tau_p / Lambda
# input_strength
alpha_p = 0.05
alpha_g = 0.05
noise_ratio = 0.007
neural_noise_sigma = 0.12
trial_num = 50
z_truth = 30
phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi


def GOP_decoding(z_t, phi_t, Ip, Ig, alpha_p_infer, alpha_g_infer, Ag, Ap, Rp, total_itenoise_rations=2000):
  sigma_g = jnp.sqrt(jnp.sqrt(np.pi) * Ag ** 3 * rho_g * tau_g / (a_g * alpha_g_infer))
  sigma_phi = jnp.sqrt(8 * jnp.pi * Ag * tau_g / (Lambda * J_pg * rho_p * Rp))
  sigma_p = jnp.sqrt(jnp.sqrt(jnp.pi) * Ap ** 3 * rho_p * tau_p / (a_p * alpha_p_infer))
  sigma_g_infer = sigma_g * noise_ratio
  sigma_phi_infer = sigma_phi * noise_ratio
  sigma_p_infer = sigma_p * noise_ratio
  eta = 5. * 1e-6
  '''
  GOP: Gradient based Optimization of Posterior
  Ip shape [n_p]
  Ig shape [M, n_g]
  '''
  z_ts = []
  phi_ts = []
  z_ts.append(z_t)
  phi_ts.append(phi_t)
  z_encode_space = jnp.linspace(0, L, num_p, endpoint=False)

  def step(carry, input):
    phi_t, z_t = carry

    fg_prime = jnp.zeros((M, num_g))
    for i in range(M):
      dis_theta = circ_dis(theta, phi_t[i])
      fg_prime = fg_prime.at[i].set(dis_theta / (2 * a_g[i] ** 2) * jnp.exp(-dis_theta ** 2 / (4 * a_g[i] ** 2)))

    dis_z = z_encode_space - z_t
    # fp = np.exp(-dis_z**2 / (4 * a_p**2))
    fp_prime = dis_z / (2 * a_p ** 2) * jnp.exp(-dis_z ** 2 / (4 * a_p ** 2))

    ## compute the log likelihood
    # partial ln P(rg|phi) / partial phi
    Ig_fgprime_prod = Ig * fg_prime  # shape [M, n_g]
    Ig_fgprime_prod = jnp.sum(Ig_fgprime_prod, axis=1)  # shape [M]
    dphi_fr = Ig_fgprime_prod / sigma_g_infer ** 2
    # print(dphi_fr)

    # partial ln P(rp|z) / partial z
    Ip_fp_prime_prod = Ip * fp_prime  # shape [n_p]
    Ip_fp_prime_prod = jnp.sum(Ip_fp_prime_prod)  # shape [1]
    dr_fr = Ip_fp_prime_prod / sigma_p_infer ** 2

    ## transition model
    phi_z = jnp.mod(z_t / Lambda, 1) * 2 * np.pi
    dis_phi = circ_dis(phi_z, phi_t)  # shape [M]
    # partial ln P(phi|z) / partial phi
    dphi_tr = 1 / sigma_phi_infer ** 2 * dis_phi  # shape [M]
    # partial ln P(phi|z) / partial z
    dr_tr = np.sum(-2 * jnp.pi / (Lambda * sigma_phi_infer ** 2) * dis_phi)
    # print(dr_tr)
    ## update
    dphi = dphi_fr + dphi_tr
    phi_t = phi_t + eta * dphi

    # boundary condition
    phi_t = jnp.mod(phi_t, 2 * np.pi)

    dr = dr_fr + dr_tr
    z_t = z_t + eta * dr

    carry = (phi_t, z_t)
    return carry, carry

  (phi_t, z_t), (phi_ts, z_ts) = bst.transform.scan(step, (phi_t, z_t), np.arange(total_itenoise_rations))
  return z_t


@jax.jit
@bst.transform.jit
def run_a_trial(key):
  bst.random.DEFAULT.seed(key)

  final_model = Coupled_Net(num_module=M)
  bst.init_states(final_model)

  z_truth = 30
  phi_truth = np.mod(z_truth / Lambda, 1) * 2 * np.pi
  z_e = z_truth
  psi = phi_truth
  Ig = jnp.zeros((M, num_g))
  for j in range(M):
    dis_theta = circ_dis(theta, psi[j])
    Ig = Ig.at[j].set(jnp.exp(-dis_theta ** 2 / (4 * a_g[j] ** 2)) + neural_noise_sigma * bst.random.randn(num_g))

  x = np.linspace(0, L, num_p, endpoint=False)
  dis_x = x - z_e
  Ip = np.exp(-dis_x ** 2 / (4 * a_p ** 2)) + neural_noise_sigma * bst.random.randn(num_p)
  z_decode_n, _, up, rp, ug, rg = final_model.Net_decoding(z_truth, phi_truth, Ip, Ig)

  max_up = jnp.max(up, axis=1)
  max_rp = jnp.max(rp, axis=1)
  Ap = jnp.max(max_up[-1])
  Rp = jnp.max(max_rp[-1])
  Ag = jnp.zeros(M)
  Rg = jnp.zeros(M, )
  for mi in range(M):
    max_ug = jnp.max(ug[:, mi, :], axis=1)
    max_rg = jnp.max(rg[:, mi, :], axis=1)
    Ag = Ag.at[mi].set(jnp.max(max_ug[-1]))
    Rg = Rg.at[mi].set(jnp.max(max_rg[-1]))
  z_decode_g = GOP_decoding(z_t=z_truth, phi_t=phi_truth, Ip=Ip, Ig=Ig,
                            alpha_p_infer=0.05, alpha_g_infer=0.05,
                            Ap=Ap, Rp=Rp, Ag=Ag)

  return z_decode_g, z_decode_n[-1]


z_decode_gop, z_decode_net = jax.vmap(run_a_trial)(bst.random.split_keys(trial_num))

z_decode_all = np.concatenate([z_decode_gop, z_decode_net])
min_z = np.min(z_decode_all)
max_z = np.max(z_decode_all)
plt.plot(z_decode_gop, z_decode_net, '.')
plt.plot([min_z, max_z], [min_z, max_z], 'k--')
plt.savefig('net_gop_comparison.png')
plt.show()
