import numpy as np
import torch
import shutil



if __name__ =='__main__':
    src = '/home/snytav/GravNN/Examples/input.txt'
    dst = '.'
    shutil.copy(src,dst)

    x = np.loadtxt('input.txt')
    x = torch.from_numpy(x)
    x.requires_grad = True

    from GAIKnet import GAIKnet
    model = GAIKnet(x.shape[0])