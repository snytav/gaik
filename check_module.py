import numpy as np
trace = True
import torch
def get_tensor_file_name(layer_name, name, subtitle, global_epoch_number):
    fn = layer_name + '_' + name + '_' + subtitle + '{:05d}'.format(global_epoch_number) + '.txt'
    return fn


def check(layer_name, name, dubious_values, global_epoch_number):
    if not trace:
        return 0.0
    else:
        fn = get_tensor_file_name(layer_name, name, '', global_epoch_number)
        correct_values = np.loadtxt(fn)
        res = np.max(np.abs(correct_values - dubious_values.numpy()))
        return res
