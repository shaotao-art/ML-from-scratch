import numpy as np
from src.base.layer import Layer

class Linear(Layer):
    def __init__(self, in_dim, out_dim) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.params = {
            'weight': None,
            'bias': None
        }

        self.gradient = {
            'd_weight': None,
            'd_bias': None
        }

        self.cache = {'x': None}

        self._init_params()

    def _init_params(self):
        self.params['weight'] = 0.01 * np.random.randn(self.in_dim, self.out_dim)
        self.params['bias'] = 0.01 * np.zeros((self.out_dim, ))

        self.gradient['d_weight'] = np.zeros_like(self.params['weight'])
        self.gradient['d_bias'] = np.zeros_like(self.params['bias'])

    def __str__(self) -> str:
        out_str = f"Linear Layer\n\
                    in_dim: {self.in_dim}, out_dim: {self.out_dim}\n\
                    #params: {self.params['weight'].size + self.params['bias'].size}\n"
        return out_str

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, f'Linear input dim should equal to in dim, get {x.shape[-1]}, expect {self.in_dim}'
        self.cache['x'] = x
        return x @ self.params['weight'] + self.params['bias']

    def backward(self, d_last):
        # compute gradient to self params
        # also need to compute gradient for next node
        assert d_last.shape[-1] == self.out_dim, f'Linear d_last dim should equal to out dim, get {d_last.shape[-1]}, expect {self.out_dim}'
        self.gradient['d_weight'] = self.cache['x'].T @ d_last
        self.gradient['d_bias'] = np.sum(d_last, axis=0)
        dx = d_last @ self.params['weight'].T
        return dx
 
