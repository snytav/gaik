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
        self.fc       = nn.Linear(3*self.N,4*self.N)
        self.cart     = Cart2_Pines_Sph_Layer()
        scale_potential = True
        # use_transition_potential = True
        min_ = 0.0
                       #ScaleNNPotential(scaler, power, R, R_min, scale_potential, min_)
        self.scale_nn = ScaleNNPotential(2,16e3,0.0,True,scale_potential, min_)
        self.analytic = AnalyticModelLayer()
        self.enf      = EnforceBoundaryConditions()
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

    def forward(self,x):

        x         = self.cart(x)
        x         = self.inv_r(x)
        x         = self.fc(x)
        x_nn,x_an = self.analytic(x)
        x         = self.scale_nn(x_nn,x_an)
        x         = self.fn(x)

        x         = self.enf(x)
        return x



if __name__ == '__main__':
    import numpy as np
    x = np.loadtxt('cart_input_00000.txt')
    model = GAIKnet(x.shape[0])

    y = model(x)

