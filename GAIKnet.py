#sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


import torch
import torch.nn as nn



from ScaleNN import ScaleNNPotential
from analytic import AnalyticModelLayer
from cart import Cart2_Pines_Sph_Layer
from inverse_r import Inv_R_Layer
from enforce import EnforceBoundaryConditions
from fuse import FuseModels
import torch.nn as nn
from check_module import check,get_tensor_file_name

from fuse import FuseModels



class GAIKnet(nn.Module):
    def __init__(self,N,trace_flag):

        super(GAIKnet,self).__init__()

        self.trace = trace_flag
        self.epoch = 0
        self.N = N
        self.inv_r    = Inv_R_Layer()
        self.fc       = nn.Linear(5*self.N,4*self.N)
        self.cart     = Cart2_Pines_Sph_Layer()
        scale_potential = True
        self.fuse = FuseModels(True)
        # use_transition_potential = True
        min_ = 0.0
                       #ScaleNNPotential(scaler, power, R, R_min, scale_potential, min_)
        self.scale_nn = ScaleNNPotential(2,16e3,0.0,True,scale_potential, min_)
        R = 16000.0
        R_min = 0.195
        mu = 0.3705464
        C20 = 0.0
        scaler = 6.25e-5
        an = AnalyticModelLayer
        self.analytic = AnalyticModelLayer(R, R_min, mu, C20, scaler)
        r_max = 3.0
        tanh_r = 3.0
        tanh_k = 0.1
        self.enf = EnforceBoundaryConditions(True, True,
                                       r_max, tanh_r, tanh_k)
        # self.enf      = EnforceBoundaryConditions()
        self.fn       = FuseModels(True)
        self.fuse_models = True


    # 0 input_1[(None, 1)][(None, 1)]
    # 1 cart2_pines_sph_layer(None, 1)(None, 2)
    # 2 inv_r_layer(None, 2)(None, 3)
    # 3 dense_2(None, 3)(None, 1)
    # 4 scale_nn_potential(None, 2)(None, 1)
    # 5 analytic_model_layer(None, 2)(None, 0)
    # 6 fuse_models(None, 1)(None, 0)
    # 7 enforce_boundary_conditions(None, 2)(None, 0)
    def remove_analytic_model(self, x, y_dict, y_hat_dict):
        if self.fuse_models:
            y_analytic_dict = self.call_analytic_model(x)
            for key in y_dict.keys() & y_analytic_dict.keys():
                y_dict[key] -= y_analytic_dict[key]
                y_hat_dict[key] -= y_analytic_dict[key]
        return y_dict, y_hat_dict

    def norm(self,y_hat):
        y = np.loadtxt('y_dict.txt')
        dy = y_hat - y
        rms = nn.RMSNorm(dy)
        return rms

    def get_layer_type(self,fname):
        name = fname.split('_')[0]


    def get_layer(self,name):
        #name = self.get_layer_type(fname)

        if name == 'cart':
            return self.cart

        if name == 'inv':
            return self.inv_r

        if name == 'analytic':
            return self.analytic

        if name == 'fuse':
            return self.fuse

        if name == 'enforce':
            return self.enf

        if name == 'dens':
            return self.fc

    def get_input_output_file(self,inout,list_of_strings):
        found_strings_comp = [string for string in list_of_strings if inout in string]
        return found_strings_comp[0]

    def layer_run_and_check(self,layer_sequence,lnum):
        lseq = layer_sequence
        layer = self.get_layer(lseq[lnum][0])

        in_file = self.get_input_output_file('input', lseq[lnum])
        in_x = torch.from_numpy(np.loadtxt(in_file))
        out = layer(in_x)
        out_file = self.get_input_output_file('output', lseq[lnum])
        out_correct = np.loadtxt(out_file)
        eps = np.max(np.abs(out_correct - out.detach().numpy()))
        return eps

    def forward(self,inputs):



        # x_nn      = self.fc(x.float())
        # M1        = int(x_nn.shape[0]/N)
        # x_nn      = x_nn.reshape(N,M1)
        # x_an      = self.analytic(x.reshape(N,5))
        # x         = self.scale_nn(x_nn,x_an)
        # #x         = self.fn(x)

        features = self.cart(inputs)
        res = check('cart','output',features)

        # N = features.shape[0]
        # M = features.shape[1]
        # x = features.reshape(N * M)
        res = check('inv_r', 'input', features)

        u_nn = self.inv_r(features)
        res = check('inv_r', 'output', u_nn)
        res = check('analytic', 'input', features)

        res = check('analytic', 'input', features)
        u_analytic = self.analytic(features)
        res = check('analytic', 'output', u_analytic.detach())

        res = check('analytic', 'output', u_analytic.detach())

        u_nn_scaled = self.scale_nn(features, u_nn)
        u_fused = self.fuse_models(u_nn_scaled, u_analytic)
        u = self.enf(features, u_fused, u_analytic)

        return u



if __name__ == '__main__':
    import numpy as np
    import os
    from file_list import list_files_by_mask,get_layer_sequence
    # x = torch.from_numpy(np.loadtxt('cart_input_00000.txt'))
    # os.remove('./cart_input_00000.txt')


    #y = model(x)

    matching_files = list_files_by_mask('.', '*put*.txt')
    for f in matching_files:
        n = len(f.split('_'))
        if n > 4:
            print(f)
            os.remove(f)
    lseq = get_layer_sequence()
    x = torch.from_numpy(np.loadtxt(matching_files[0]))
    model = GAIKnet(x.shape[0], True)
    # layer = model.get_layer(lseq[0][0])
    #
    # in_file = model.get_input_output_file('input',lseq[0])
    # in_x = torch.from_numpy(np.loadtxt(in_file))
    # out = layer(in_x)
    # out_file = model.get_input_output_file('output',lseq[0])
    # out_correct = np.loadtxt(out_file)
    # eps = np.max(np.abs(out_correct-out.detach().numpy()))
    eps = model.layer_run_and_check(lseq,0)







    qq = 0

