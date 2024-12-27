import torch.nn as nn
import torch

# 6.25e-05,6.25e-05,6.25e-05
def transform(data,scaler,scale_,min_):
    # Some older networks will load an old version of uniform scaler
    # that didn't have scaler attribute. Therefore, add it if not present


    if scaler is not None:
        X = data * scaler
    else:
        X = data * scale_ + min_
    return X

# 16000, 0.0
def compute_shape_parameters(R,R_min):

    R_vec = torch.tensor([R, R_min, 0])
    R_vec_ND = transform(R, R_min, 0)

    a = transform(R, R_min, 0,torch.tensor([0, 0]))
    b = transform(R, R_min, 0,torch.tensor([0, 1]))

    e = torch.sqrt(1 - b**2 / a**2)
    return a, b, e

def r_safety_set(r, clip=1.0):
    r_inv = torch.divide(.0, r)
    r_inv_cap = torch.clip(r_inv, 0.0, clip)
    r_cap = torch.clip(r, 0.0, clip)
    return r_cap, r_inv_cap

class ScaleNNPotential(nn.Module):
    def __init__(self,power,R,R_min,scale_potential):
        super(ScaleNNPotential, self).__init__()
        self.N = N
        self.power = power
        self.use_transition_potential = True # kwargs.get("use_transition_potential", [True])[0]

        self.scale_potential = True
        a, b, e = compute_shape_parameters(R, R_min)
        self.a = a
        self.b = b
        self.e = e

    def call(self, features, u_nn):
        r = features[:, 0:1]
        r_cap, r_inv_cap = r_safety_set(r)

        if not self.scale_potential:
            return u_nn

        # scale the internal potential
        # scale = internal_scale(r_cap, self.a)
        # u_scaled_internal = tf.negative(u_nn * scale)

        # scale the external potential down to correct order of mag
        # U = U_NN * 1 / r^power
        scale_external = torch.pow(r_inv_cap, self.power)

        # Don't scale within the critical radius (1+e)
        tf.ones_like(scale_external)

        # Must use a smooth_step function instead of tanh to
        # force solution scaling to 0 or 1.
        # R_trans = 1.0 + self.e
        # scale = blend_smooth(r, scale_internal, scale_external, R_trans, 2*R_trans)
        # u_final = u_nn * scale
        u_final = u_nn * scale_external

        return u_final