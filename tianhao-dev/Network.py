import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax

bm.set_dt(0.2) #length of time step
bm.set_platform('gpu')
# bm.set_platform('gpu')

class Place_net(bp.DynamicalSystem):
    def __init__(self, z_min, z_max, num=1000, k = 1.,
                 tau=10.,  a_p = 4., J0=50.):
        super(Place_net, self).__init__()

        # parameters
        self.tau = tau  
        self.k = k # Global inhibition
        self.a = a_p  # Range of excitatory connections
        self.J0 = J0  # maximum connection value
        self.num = num
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, num, endpoint = False)  # The encoded feature values
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density
        # variables
        self.r = bm.Variable(bm.zeros(num))
        self.r_mec = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1))
        self.center_input = bm.Variable(bm.zeros(1))
        # Connections
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat) # recurrent conn
        
    def period_bound(self, A):
        B = bm.where(A > self.z_range/2, A - self.z_range, A)
        B = bm.where(B < -self.z_range/2, B + self.z_range, B)
        return B
    
    def make_conn(self):
        d = self.period_bound(bm.abs(self.x[0] - self.x))
        Jxx = self.J0/(self.a*np.sqrt(2*np.pi)) * bm.exp(-0.5 * bm.square(d / self.a)) 
        return Jxx
    
    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))

    def get_center(self):
        r = self.r
        # r0 = self.r
        # r = bm.where(self.r>bm.max(r0)/10, self.r, 0)
        x = np.linspace(-np.pi,np.pi,self.num,endpoint=False)
        exppos = bm.exp(1j * x)
        self.center[0] = (bm.angle(bm.sum(exppos * r))/2/np.pi+1/2) * self.z_range + self.z_min

    def update(self, Ip, I_grid):
        # Calculate self position
        self.get_center()
        # Calculate recurrent input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))  # real and abs
        # Calculate total input
        self.input.value = I_grid + Irec + Ip
        # Update neural state
        du = (-self.u + self.input) / self.tau * bm.dt
        u = self.u + du
        self.u.value = bm.where(u > 0, u, 0)
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2

class Grid_net(bp.DynamicalSystem):
    def __init__(self, L, z_min, z_max, num=100, num_hpc=1000, k_mec=1.,
                 tau=1.,  a_g=1., J0=50., W0 = 0.1):
        super(Grid_net, self).__init__()

        # parameters
        self.tau = tau  
        self.k = k_mec   # Global inhibition
        self.L = L # Spatial Scale
        self.a = a_g # Range of excitatory connections
        self.J0 = J0  # maximum connection value
        self.W0 = W0
        self.num = num
        self.num_hpc = num_hpc
        # feature space
        self.x_hpc = np.linspace(z_min, z_max, num_hpc, endpoint=False)  # The encoded feature values
        self.x = np.linspace(-np.pi, np.pi, num, endpoint=False)  # The encoded feature values
        self.rho = num / np.pi/2  # The neural density
        self.dx = np.pi*2  / num  # The stimulus density
        
        # variables
        self.r = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1)) 
        self.center_input = bm.Variable(bm.zeros(1))
        # Connections
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat) # Grid cell recurrent conn
        self.conn_out = self.make_conn_out() # From grid cells to place cells

    def xtopi(self,x):
        return (x%self.L)/self.L*2*np.pi-np.pi
        
    def period_bound(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B
    
    def make_conn_out(self):
        theta_hpc = self.xtopi(self.x_hpc)
        D = theta_hpc[:, None] - self.x
        Dis_circ = self.period_bound(D)
        conn_out = self.W0/(self.a*np.sqrt(2*np.pi)) * bm.exp(-0.5 * bm.square(Dis_circ / self.a))
        return conn_out
    
    def make_conn(self):
        d = self.period_bound(bm.abs(self.x[0] - self.x))
        Jxx = self.J0/(self.a*np.sqrt(2*np.pi)) * bm.exp(-0.5 * bm.square(d / self.a)) 
        return Jxx

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))
        # self.center.value = 2 * bm.pi * bm.random.rand(1) - bm.pi

    def get_center(self):
        exppos = bm.exp(1j * self.x)
        self.center[0] = bm.angle(bm.sum(exppos * self.r))
        self.center_input[0] = bm.angle(bm.sum(exppos * self.input))

    def update(self, Ig, r_hpc):
        # Calculate self position
        self.get_center()
        # Calculate hpc input
        conn_in = bm.transpose(self.conn_out)
        input_hpc = bm.matmul(conn_in, r_hpc)
        # Calculate recurrent input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))  # real and abs
        # Calculate total input
        self.input.value = input_hpc + Irec + Ig
        # Update neural state
        du = (-self.u + self.input) / self.tau * bm.dt
        u = self.u + du
        self.u.value = bm.where(u > 0, u, 0)
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2

class Coupled_Net(bp.DynamicalSystemNS):
    def __init__(self, HPC_model, MEC_model_list, num_module):
        super(Coupled_Net, self).__init__()

        self.HPC_model = HPC_model
        self.MEC_model_list = MEC_model_list
        self.num_module = num_module
        self.x_hpc = HPC_model.x
        self.a_p = HPC_model.a
        self.num_hpc = HPC_model.num
        self.I_mec = bm.zeros([self.num_hpc, ])
        self.num_hpc = HPC_model.num
        self.W_G = bm.ones([self.num_module,])
        self.phase = bm.Variable(bm.zeros([self.num_module,]))
        self.energy = bm.Variable(bm.zeros(1,))
        # self.W_G = bm.Variable(bm.ones([self.num_module,]))
    def reset_state(self):
        self.HPC_model.reset_state()
        for MEC_model in self.MEC_model_list:
            MEC_model.reset_state()

    def initial(self, alpha_p, alpha_g, Ip, Ig):
        # Update MEC states
        r_hpc = bm.zeros(self.HPC_model.num)
        I_mec = bm.zeros(self.HPC_model.num)
        i = 0
        for MEC_model in self.MEC_model_list:
            MEC_model.update(alpha_g*Ig[i], r_hpc)
            i+=1
        # Update Hippocampus states
        self.HPC_model.update(alpha_p*Ip, I_grid = I_mec)

    def Get_Energy(self, alpha_p, alpha_g, Ip, Ig):
        ug = self.MEC_model_list[0].u
        up = self.HPC_model.u
        rg = self.MEC_model_list[0].r
        rp = self.HPC_model.r
        # Bump height
        Ag = bm.max(ug)
        Ap = bm.max(up)
        Rg = bm.max(rg)
        Rp = bm.max(rp)
        # Parameters
        rho_g = self.MEC_model_list[0].rho
        tau_g = self.MEC_model_list[0].tau
        tau_p = self.HPC_model.tau
        rho_p = self.HPC_model.rho
        a_p = self.HPC_model.a
        # Sigma
        Lambda = bm.zeros(self.num_module,)
        sigma_phi  = bm.zeros(self.num_module)
        for i in range(self.num_module):
            J_pg = self.MEC_model_list[i].W0
            Lambda[i] = self.MEC_model_list[i].L
            sigma_phi[i] = 1/((Lambda[i]/2/bm.pi) * bm.sqrt(J_pg*rho_g*Rg/(4*Ap*tau_p))) 
        sigma_p = bm.sqrt(bm.sqrt(bm.pi)*Ap**3*rho_p*tau_p/(a_p*alpha_p)) 
        # Feature space
        place_x = self.HPC_model.x
        theta = self.MEC_model_list[0].x
        
        # Network decoding
        z = self.HPC_model.center[0]
        phi = bm.zeros(self.num_module,)
        psi_z = bm.mod(z / Lambda, 1) * 2 * bm.pi
        for i in range(self.num_module):
            phi[i] = self.MEC_model_list[i].center[0]
        # 圆周距离函数
        def circ_dis(phi_1, phi_2):
            dis = phi_1 - phi_2
            dis = bm.where(dis>bm.pi, dis-2*bm.pi, dis)
            dis = bm.where(dis<-bm.pi, dis+2*bm.pi, dis)
            return dis
        # Calculate log posterior
        log_prior = 0
        log_likelihood_grid = 0
        for i in range(self.num_module):
            a_g = self.MEC_model_list[i].a
            sigma_g = bm.sqrt(bm.sqrt(bm.pi)*Ag**3*rho_g*tau_g/(a_g*alpha_g)) 
            dis_1 = circ_dis(theta, phi[i])
            fg = bm.exp(-dis_1**2 / (4 * a_g**2))
            log_likelihood_grid -= bm.sum((Ig[i, :] - fg)**2) / sigma_g**2
            dis_2 = circ_dis(phi[i], psi_z[i])
            log_prior -= 1 / (sigma_phi[i]**2) * bm.exp(-dis_2**2/8/a_g**2) * dis_2**2
        dis_x = place_x - z
        fp = bm.exp(-dis_x**2 / (4 * a_p ** 2))
        log_likelihood_place = -bm.sum((Ip - fp)**2) / sigma_p**2
        log_posterior = log_likelihood_grid + log_prior + log_likelihood_place
        # Energy function
        self.energy[0] = - log_posterior


    def update(self, alpha_p, alpha_g, Ip, Ig):
        self.Get_Energy(alpha_p, alpha_g, Ip, Ig)
        # Update MEC states
        r_hpc = self.HPC_model.r
        I_mec_module = bm.zeros([self.num_hpc, self.num_module])
        i = 0
        for MEC_model in self.MEC_model_list:
            r_mec = MEC_model.r
            I_basis = bm.matmul(MEC_model.conn_out, r_mec)
            I_mec_module[:,i] = I_basis 
            MEC_model.update(alpha_g*Ig[i], r_hpc)
            self.phase[i] = MEC_model.center[0]
            i+=1
        self.I_mec = bm.matmul(I_mec_module, self.W_G)
        # Update Hippocampus states
        self.HPC_model.update(alpha_p*Ip, I_grid = self.I_mec)
