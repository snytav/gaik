import torch
import torch.nn as nn

class Cart2_Pines_Sph_Layer(nn.Module):
    def __init__(self):
        super(Cart2_Pines_Sph_Layer,self).__init__()

    def forward(self,x):
        r = torch.norm(x,dim=1)
        stu = x/r.unsqueeze(1)
        spheres = torch.cat((r.unsqueeze(1),stu),1)
        return spheres

import numpy as np

if __name__ == '__main__':
    x = np.loadtxt('cart_inputs.txt')
    x = torch.from_numpy(x)
    car = Cart2_Pines_Sph_Layer()
    y = car(x)

    qq = 0

