#import tensorflow



from ScaleNN import ScaleNNPotential
from analytic import AnalyticModelLayer

class GAIKnet(nn.Module):
    def __init__(self,N):
        super(GAIKnet,self).__init__()
        self.N = N
        self.inv_r = Inv_R_Layer()
        return
        self.fc = nn.Linear(3*self.N,4*self.N)
        self.cart = Cart2_Pines_Sph_Layer()
        self.scale_nn = ScaleNNPotential(2,16e3,0.0,True)
       # self.analytic = AnalyticModelLayer()

    # 0 input_1[(None, 1)][(None, 1)]
    # 1 cart2_pines_sph_layer(None, 1)(None, 2)
    # 2 inv_r_layer(None, 2)(None, 3)
    # 3 dense_2(None, 3)(None, 1)
    # 4 scale_nn_potential(None, 2)(None, 1)
    # 5 analytic_model_layer(None, 2)(None, 0)
    # 6 fuse_models(None, 1)(None, 0)
    # 7 enforce_boundary_conditions(None, 2)(None, 0)

    def forward(self,x):
        #x = self.cart(x)
        x = self.inv_r(x)
        # x = self.fc(x)
        # x = self.scale_nn(x)
        # x = self.analytic(x)
        return x



if __name__ == '__main__':
    import numpy as np
    x = np.loadtxt('input.txt')
    model = GAIKnet(x.shape[0])

    y = model(x)

