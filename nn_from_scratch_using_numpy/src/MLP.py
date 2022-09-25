from src.np_MLP_layers import Linear, ReLU
import numpy as np


class MLP:
    """
    MLP model without batchnorm
    params:
        dim_lst: [in_dim, ..., out_dim] for linear regression the last element of dim_lst must be 1
    """
    def __init__(self, dim_lst:list) -> None:
        assert dim_lst[-1] == 1
        self.dim_lst = dim_lst

        self.layers = []
        for i in range(len(self.dim_lst) - 2):
            self.layers.append(Linear(self.dim_lst[i], self.dim_lst[i + 1]))
            self.layers.append(ReLU())
        self.layers.append(Linear(self.dim_lst[-2], 1))


    def __str__(self) -> str:
        out_str = f'this is a MLP\n'
        for layer in self.layers:
            out_str += str(layer)
        return out_str 
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_last):
        for layer in reversed(self.layers):
            d_last = layer.backward(d_last)
    
    def update_params(self, l_r):
        for layer in self.layers:
            layer.update_params(l_r)