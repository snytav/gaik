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

class GAIKnet(nn.Module):
    def __init__(self,N):
        super(GAIKnet,self).__init__()
        self.N = N
        self.inv_r    = Inv_R_Layer()
        self.fc       = nn.Linear(5*self.N,4*self.N)
        self.cart     = Cart2_Pines_Sph_Layer()
        scale_potential = True
        self.fuse =
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

    def forward(self,inputs):



        # x_nn      = self.fc(x.float())
        # M1        = int(x_nn.shape[0]/N)
        # x_nn      = x_nn.reshape(N,M1)
        # x_an      = self.analytic(x.reshape(N,5))
        # x         = self.scale_nn(x_nn,x_an)
        # #x         = self.fn(x)

        features = self.cart(inputs)
        # N = features.shape[0]
        # M = features.shape[1]
        # x = features.reshape(N * M)
        u_nn = self.inv_r(features)
        u_analytic = self.analytic(features)
        u_nn_scaled = self.scale_nn(features, u_nn)
        u_fused = self.fuse_models(u_nn_scaled, u_analytic)
        u = self.enf(features, u_fused, u_analytic)

        return u



if __name__ == '__main__':
    import numpy as np
    x = torch.from_numpy(np.loadtxt('cart_input_00000.txt'))
    model = GAIKnet(x.shape[0])

    y = model(x)
    qq = 0

