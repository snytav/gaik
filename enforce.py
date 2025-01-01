import torch
import torch.nn as nn
import numpy as np


def H(x, r, k):
    # assuming k > 0 this goes from 0 -> 1 as x->inf
    dr = torch.subtract(x, r)
    h = 0.5 + 0.5 * torch.tanh(k * dr)
    return h


def G(x, r, k):
    # assuming k > 0 this goes from 1 -> 0 as x->inf
    return 1.0 - H(x, r, k)


class EnforceBoundaryConditions(nn.Module):
    def __init__(self,enforce_bc,trainable_tanh,r_max,tanh_r,tanh_k): # **kwargs):
        super(EnforceBoundaryConditions, self).__init__()
        self.enforce_bc = enforce_bc # kwargs["enforce_bc"][0]
        #kwargs.get("ref_radius_analytic", [np.nan])[0]
        self.trainable_tanh = trainable_tanh #kwargs.get("trainable_tanh")[0]
        self.r_max = tanh_r #kwargs.get("tanh_r", [r_max])[0]
        self.k_init = tanh_k #kwargs.get("tanh_k", [1.0])[0]

        self.radius = nn.Parameter(torch.tensor(self.r_max))

        self.k = nn.Parameter(torch.tensor(self.k_init))

    def forward(self, features, u_nn, u_analytic):
        if not self.enforce_bc:
            return u_nn
        r = features[:, 0:1]
        h = H(r, self.radius, self.k)
        g = G(r, self.radius, self.k)
        u_model = g * u_nn + h * u_analytic
        return u_model



if __name__ == '__main__':
    r_max = 3.0
    tanh_r = 3.0
    tanh_k = 0.1
    bc = EnforceBoundaryConditions(True,True,
                                   r_max,tanh_r,tanh_k)

    features = torch.from_numpy(np.loadtxt('enforce_features.txt'))
    u_nn = torch.from_numpy(np.loadtxt('enforce_u_nn.txt'))
    u_analytic = torch.from_numpy(np.loadtxt('enforce_u_analytic.txt'))
    y = bc(features,u_nn,u_analytic)

