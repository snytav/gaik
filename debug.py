import numpy as np
import torch.nn as nn


class DebugLayer(nn.Module):
      def __init__(self):
          super(DebugLayer,self).__init__()
          self.debug_flag = True
          self.layer_name = 'debug_default'
          self.path = '/home/snytav/PycharmProjects/GravNN/Examples/'
          self.n_epoch = 0

      def next_epoch(self):
          self.n_epoch = self.n_epoch + 1

      def read_array(self,array_name,arr):
          if len(arr.shape) == 2 and arr.shape[1] == 1:
              arr = arr.reshape(arr.shape[0]*arr.shape[1])
          if self.debug_flag:
             fname = self.layer_name + '_' + array_name + '_' + '{:05d}'.format(self.n_epoch) + '.txt'
             p =  np.loadtxt(self.path+fname)
             eps = np.max(np.abs(p - arr.detach().numpy()))
             return eps
          else:
             return 1.0


