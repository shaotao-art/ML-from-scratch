import numpy as np
from src.base.layer import Layer
from src.base.layer import Params


class Linear(Layer):
    def __init__(self, in_dim, out_dim) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.params = {
            'weight': Params(),
            'bias': Params()
        }

        self.cache = {'x': None}

        self._init_params()

    def _init_params(self):
        # Initialize the weights
        limit = 1 / np.sqrt(self.in_dim)
        self.params['weight'].data = np.random.uniform(-limit, limit, (self.in_dim, self.out_dim))
        self.params['bias'].data= np.zeros((self.out_dim, ))

        self.params['weight'].gradient = np.zeros_like(self.params['weight'].data)
        self.params['bias'].gradient = np.zeros_like(self.params['bias'].data)

    def __str__(self) -> str:
        out_str = f"Linear Layer\n\
                    in_dim: {self.in_dim}, out_dim: {self.out_dim}\n\
                    #params: {self.params['weight'].data.size + self.params['bias'].data.size}\n"
        return out_str

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, f'Linear input dim should equal to in dim, get {x.shape[-1]}, expect {self.in_dim}'
        self.cache['x'] = x
        return x @ self.params['weight'].data + self.params['bias'].data

    def backward(self, d_last):
        # compute gradient to self params
        # also need to compute gradient for next node
        assert d_last.shape[-1] == self.out_dim, f'Linear d_last dim should equal to out dim, get {d_last.shape[-1]}, expect {self.out_dim}'
        self.params['weight'].gradient = self.cache['x'].T @ d_last
        self.params['bias'].gradient = np.sum(d_last, axis=0)
        dx = d_last @ self.params['weight'].data.T
        return dx
 
