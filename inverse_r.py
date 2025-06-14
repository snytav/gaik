import torch
import torch.nn as nn
import numpy as np


def r_safety_set(r, clip=1.0):
    r_inv = torch.divide(torch.ones_like(r), r)
    r_inv_cap = torch.clip(r_inv, 0.0, clip)
    r_cap = torch.clip(r, 0.0, clip)
    return r_cap, r_inv_cap

class Inv_R_Layer(nn.Module):
    def __init__(self):
        super(Inv_R_Layer,self).__init__()
    def forward(self,inputs):
        r = inputs[:, 0:1]
        r_cap, r_inv_cap = r_safety_set(r)
        spheres = torch.concat([r_cap, r_inv_cap, inputs[:, 1:4]], axis=1)
        return spheres[:,:3]


if __name__ == '__main__':
    ir = Inv_R_Layer()
    x = np.loadtxt('in_r_inputs.txt')
    x = torch.from_numpy(x)

    y = ir(x)
    qq = 0
