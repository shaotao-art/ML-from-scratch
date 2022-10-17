from src.base.layer import Layer
import numpy as np

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.cache = {
            'x': None
        }


    def __str__(self) -> str:
        out_str = 'ReLU layer\n'
        return out_str

    def forward(self, x):
        self.cache['x'] = x
        x[x < 0] = 0
        return x

    def backward(self, d_last):
        filt = self.cache['x'] < 0
        d_last[filt] = 0
        return d_last