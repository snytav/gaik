import numpy as np
import torch
import torch.nn as nn


class FuseModels(nn.Module):
    def __init__(self,fuse_models):
        super(FuseModels,self).__init__()
        self.fuse = fuse_models

    def forward(self, u_nn, u_analytic):
        fuse_vector = int(self.fuse)
        u = u_nn + fuse_vector * u_analytic
        return u



if __name__ == '__main__':
    fm = FuseModels(True)
    x_nn = np.loadtxt('fuse_in_u_nn')
    x_an = np.loadtxt('fuse_in_u_an.txt')
    y = fm(x_nn,x_an)