import numpy as np
import torch
import torch.nn as nn
from debug import DebugLayer


class FuseModels(DebugLayer):
    def __init__(self,fuse_models):
        super(FuseModels,self).__init__()
        self.fuse = fuse_models
        self.layer_name = 'fuse'

    def forward(self, u_nn, u_analytic):
        u_nn = u_nn.reshape(u_nn.shape[0],1)

        e1 = self.read_array('input_u_analytic',u_analytic)
        e2 = self.read_array('input_u_nn',u_nn)
        fuse_vector = int(self.fuse)
        u = u_nn + fuse_vector * u_analytic
        e = self.read_array('output',u)
        return u



if __name__ == '__main__':
    fm = FuseModels(True)
    x_nn = np.loadtxt('fuse_in_u_nn')
    x_an = np.loadtxt('fuse_in_u_an.txt')
    y = fm(x_nn,x_an)