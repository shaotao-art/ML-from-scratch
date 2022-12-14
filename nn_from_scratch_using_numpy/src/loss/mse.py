import numpy as np
from src.base.layer import Layer


class MSE(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.cache = {
            'x': None,
            'y': None
        }

    def __str__(self) -> str:
        out_str = 'this is a MSE layer'
        return out_str

    def forward(self, x, y):
        b_s, _ = x.shape
        self.cache['x'] = x
        self.cache['y'] = y
        loss = np.sum((x - y) ** 2) / b_s
        return loss

    def backward(self):
        n, _ = self.cache['x'].shape
        return 2 * (self.cache['x'] - self.cache['y']) / n
