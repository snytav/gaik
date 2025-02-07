import torch
import torch.nn as nn
from debug import DebugLayer

class Cart2_Pines_Sph_Layer(DebugLayer):
    def __init__(self):
        super(Cart2_Pines_Sph_Layer,self).__init__()
        self.layer_name = 'cart'

    def forward(self,x):
        eps1 = self.read_array('input',x)
        r = torch.norm(x,dim=1)
        stu = x/r.unsqueeze(1)
        spheres = torch.cat((r.unsqueeze(1),stu),1)
        eps2 = self.read_array('output',spheres)
        return spheres

import numpy as np

if __name__ == '__main__':
    x = np.loadtxt('cart_inputs.txt')
    x = torch.from_numpy(x)
    car = Cart2_Pines_Sph_Layer()
    y = car(x)

    qq = 0

