import numpy as np
trace = True
epoch = 0
import torch
def get_tensor_file_name(layer_name, name, subtitle, global_epoch_number):
    fn = layer_name + '_' + name + '_' + subtitle + '{:05d}'.format(global_epoch_number) + '.txt'
    return fn


def check(layer_name, name, dubious_values):
    if not trace:
        return 0.0
    else:
        if len(dubious_values.shape) > 1 and dubious_values.shape[1] == 1:
            dubious_values = dubious_values.reshape(dubious_values.shape[0])

        fn = get_tensor_file_name(layer_name, name, '', epoch)
        correct_values = np.loadtxt(fn)
        res = np.max(np.abs(correct_values - dubious_values.numpy()))
        return res
