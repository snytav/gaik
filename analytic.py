import torch
import torch.nn as nn
import numpy as np





def H(x, r, k):
    # assuming k > 0 this goes from 0 -> 1 as x->inf
    dr = torch.subtract(x, r)
    h = 0.5 + 0.5 * torch.tanh(k * dr)
    return h


class AnalyticModelLayer(nn.Module):
    def __init__(self, R,R_min,mu,C20,scaler):
        super(AnalyticModelLayer, self).__init__()
        self.layer_name = 'analytical'

        # defaults to zero
        self.mu = 0.0#kwargs.get("mu_non_dim", [0.0])[0]
        self.C20 = C20
        self.scaler = scaler

        # self.use_transition_potential = transition_potential

        self.c1 = np.sqrt(15.0 / 4.0) * np.sqrt(3.0)
        self.c2 = np.sqrt(5.0 / 4.0)

        from ScaleNN import compute_shape_parameters
        a, b, e = compute_shape_parameters(R,R_min,R_min,scaler)
        self.a = a
        self.b = b

        self.mu = mu

        self.r_external = torch.nn.Parameter(torch.zeros(1))
        self.k_external = torch.nn.Parameter(0.5*torch.ones(1))

        self.trainable_tanh = True

    def forward(self, inputs):
        r = inputs[:, 0:1]
        u = inputs[:, 3:4]

        from ScaleNN import r_safety_set
        r_cap, r_inv_cap = r_safety_set(r)

        # External
        # Compute point mass approximation assuming
        u_pm_external = self.mu * r_inv_cap
        eps = self.read_array('u_pm_external',u_pm_external)
        u_C20 = (
            (self.a * r_inv_cap) ** 2
            * u_pm_external
            * (u**2 * self.c1 - self.c2)
            * self.C20
        )
        u_external_full = torch.neg(u_pm_external + u_C20)
        epsf = self.read_array('u_pm_external_full', u_external_full)
        # Internal
        u_external_pm_boundary = self.mu / self.a
        u_external_C20_boundary = (
            u_external_pm_boundary * (u**2 * self.c1 - self.c2) * self.C20
        )
        u_boundary = torch.neg(u_external_pm_boundary + u_external_C20_boundary)
        u_internal = self.mu * (r_cap**2 / self.a**3) + 2 * u_boundary

        u_analytic = torch.where(r < self.a, u_internal, u_external_full)

        # decrease the weight of the model in the region between
        # the interior of the asteroid and out to r < 1 + e, where
        # e is the eccentricity of the asteroid geometry, because
        # in this regime, a point mass / SH assumption adds unnecessary
        # error.
        h_external = H(r, self.r_external, self.k_external)
        u_analytic = u_analytic * h_external

        return u_analytic

if __name__ == '__main__':
    R                    = 16000.0
    R_min                = 0.195
    mu                   = 0.3705464
    C20                  = 0.0
    scaler               = 6.25e-5
    an = AnalyticModelLayer(R,R_min,mu,C20,scaler)
    import numpy as np
    x = torch.from_numpy(np.loadtxt('analytic_input_00000.txt'))
    y = an(x)
    qq = 0
