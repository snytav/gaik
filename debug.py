import numpy as np
import torch.nn as nn


class DebugLayer(nn.Module):
      def __init__(self):
          self.debug_flag = True

      def write_array(layer_name,array_name, n_epoch,arr,debug_flag):
          if debug_flag:
             fname = layer_name + '_' + array_name + '_' + '{:05d}'.format(n_epoch) + '.txt'
             p =  np.savetxt(fname)
             eps = np.max(np.abs(p - arr.detach().numpy()))
             return eps
          else:
             return 1.0


