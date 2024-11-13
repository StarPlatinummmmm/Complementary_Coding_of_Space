
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os,sys

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from params import params_prob, params_LSC, params_PSC
else:
    from utils.params import params_prob, params_LSC, params_PSC

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class LSC(nn.Module):
    def __init__(self, params_LSC):
        super(LSC, self).__init__()
        self.n_p = params_LSC['n_p']
        self.a_p = params_LSC['a_p']
        self.rho_p = params_LSC['rho_p']
        self.L = params_LSC['L']
        self.R_p = 1.
        self.sigma_p = float(params_LSC['sigma_p'])
        # self.mu_p = torch.linspace(0, self.L, self.n_p + 1)[:-1] + 0.5 * self.L / self.n_p
        self.mu_p = torch.linspace(0, self.L, self.n_p + 1)[:-1]
        self.mu_p = self.mu_p.to(device)

    def forward(self, x, rate_ampl= 1.0, noiseFlag=False):
        activation_p = self.R_p * torch.exp(-0.5 * ((self.mu_p - x)**2) / (math.sqrt(2)*self.a_p)**2)
        if noiseFlag:
            sigma_p = rate_ampl * self.sigma_p
            fr_noise = torch.normal(0, sigma_p, size=(self.n_p,)).to(device)
        else:
            fr_noise = 0.0
        # print(sigma_p)
        activation_p += self.R_p*fr_noise
        # activation_p = torch.clamp(activation_p, min=0)
        return activation_p

    def forward_prime(self, x):# phi is a vector
        # prime means the derivative of the activation function
        distance = self.mu_p - x
        r_p_prime = self.R_p*torch.exp(-0.5 * distance**2 / (math.sqrt(2)*self.a_p)**2) * distance/ (math.sqrt(2) * self.a_p)**2
        return r_p_prime 
    
    def forward_batch(self, x):
        '''
        x [batch_size]
        '''
        bsz = x.shape[0]
        r_p = self.R_p*torch.exp(-0.5 * (self.mu_p - x[:, None])**2 / (math.sqrt(2)*self.a_p)**2)
        return r_p # shape [batch_size, n_p]


class PSC(nn.Module):
    def __init__(self, params_PSC):
        super(PSC, self).__init__()
        self.Lphase = params_PSC['Lphase']
        self.n_g = params_PSC['n_g']
        self.M = params_PSC['M']
        self.n_g_all = self.n_g * self.M
        self.a_gs = params_PSC['a_gs']
        self.lambda_gs = params_PSC['lambda_gs']
        self.R_g = params_PSC['R_g']
        self.sigma_g = params_PSC['sigma_g']
        self.sigma_phi = params_PSC['sigma_phi']
        # self.mu_g = torch.linspace(0, self.Lphase, self.n_g + 1)[:-1] + 0.5 * self.Lphase / self.n_g
        self.mu_g = torch.linspace(0, self.Lphase, self.n_g + 1)[:-1]
        self.mu_g = self.mu_g.to(device)
        
    def dist(self, phi_1,phi_2):
        d = phi_1 - phi_2
        d = torch.where(d > math.pi, d - 2 * math.pi, d)
        d = torch.where(d < -math.pi, d + 2 * math.pi, d)
        return d

    def forward(self, x, noiseFlag=False, rate_ampl= 1.0, phi_ampl=1.0):
        # activation_gs = torch.zeros(self.n_g_all, dtype=torch.float32)
        activation_gs = torch.zeros([self.M, self.n_g], dtype=torch.float32)
        for i in range(self.M):
            simga_phi = self.sigma_phi[i]
            sigma_g = self.sigma_g[i]
            if noiseFlag:
                simga_phi = phi_ampl * simga_phi
                sigma_g = rate_ampl * sigma_g
                phase_noise = torch.normal(0, simga_phi, size=(1,)).to(device)
                fr_noise = torch.normal(0, sigma_g, size=(self.n_g,)).to(device)
            else:
                phase_noise = 0.
                fr_noise = 0.
            
            # print(phase_noise)

            phase = x % self.lambda_gs[i] * self.Lphase / self.lambda_gs[i] + phase_noise
            # phase = (x / self.lambda_gs[i] * self.Lphase + phase_noise) % self.Lphase
            
            distance = self.dist(self.mu_g, phase)
            activation_gs[i] = 1.*torch.exp(-0.5 * distance**2 / (math.sqrt(2)*self.a_gs[i])**2) + fr_noise
            # activation_gs[i] = self.R_g[i] * activation_gs[i]
        # activation_gs = torch.clamp(activation_gs, min=0)
        return activation_gs 
    
    def forward_one_module(self, phi, module): 
        '''
        phi is a scalar
        '''
        a_g = self.a_gs[module]
        distance = self.dist(self.mu_g,phi)
        r_g = 1.* torch.exp(-0.5 * (distance**2) / (math.sqrt(2)*a_g)**2)
        # r_g = self.R_g[module]*r_g
        # r_g = torch.clamp(activation_g, min=0)
        return r_g # shape [n_g]

    def forward_one_module_batch(self, phi, module): 
        '''
        phi [phi_candidates]
        '''
        a_g = self.a_gs[module]
        distance = self.dist(self.mu_g, phi[:, None])
        r_g = 1.*torch.exp(-0.5 * (distance**2) / (math.sqrt(2)*a_g)**2)
        # r_g = self.R_g[module]*r_g
        # r_g = torch.clamp(activation_g, min=0)
        return r_g  # shape [phi_candidates, n_g]
    
    def forward_modules_batch(self, phi):
        '''
        phi [batch_size, M]
        '''
        bsz = phi.shape[0]
        r_gs = torch.zeros([bsz, self.M, self.n_g], dtype=torch.float32)
        for i in range(self.M):
            phase = phi[:, i] # shape [batch_size]
            distance = self.dist(self.mu_g, phase[:, None]) # shape [batch_size, n_g]
            r_gs[:, i, :] = 1.*torch.exp(-0.5 * distance**2 / (math.sqrt(2)*self.a_gs[i])**2)
            # r_gs[:, i, :] = self.R_g[i]*r_gs[:, i, :]
        # r_gs = torch.clamp(r_gs, min=0)
        return r_gs 

    def forward_modules_prime(self, phi):# phi is a vector
        # prime means the derivative of the activation function
        # r_gs_prime = torch.zeros(self.n_g_all, dtype=torch.float32)
        r_gs_prime = torch.zeros([self.M, self.n_g], dtype=torch.float32)
        for i in range(self.M):
            phase = phi[i]
            distance = self.dist(self.mu_g, phase)
            # r_gs_prime[i * self.n_g:(i + 1) * self.n_g] = self.R_g*torch.exp(-0.5 * distance**2 / (math.sqrt(2)*self.a_gs[i])**2) * distance/ (math.sqrt(2) * self.a_gs[i])**2
            r_gs_prime[i] = 1.*torch.exp(-0.5 * distance**2 / (math.sqrt(2)*self.a_gs[i])**2) * distance/ (math.sqrt(2) * self.a_gs[i])**2
            # r_gs_prime[i] = self.R_g[i]*r_gs_prime[i]
        return r_gs_prime 