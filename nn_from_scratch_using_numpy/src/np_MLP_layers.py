import numpy as np

"""
宏观来看，决定模块的初始化所需的参数：
    如对于Linear层
        input feature dim
        output feature dim

1. 参数管理
    prams: 存储参数
    cache: 存储用于计算梯度的中间变量
    gradient: 存储梯度

2. init_params
    初始化参数

2. forward
    前向过程 以Linear层为例
    out = input feature @ weight + bias
    (b_s, in_dim) @ (in_dim, out_dim) + out_dim = (b_s, out_dim)

3. backward
    需要计算梯度
    以Linear层为例
    input: d_last 上游传过来的累计梯度 (b_s, out_dim)
    out:
        d_w: (in_dim, out_dim), x.T @ d_last
        d_b: (out_dim), 
        d_x: (b_s, in_dim), d_last @ weight.T 

4. update_params
    根据计算出的梯度使用梯度下降法更新参数
"""


class Layer:
    """
    base class for layers
    """
    def __init__(self) -> None:
        pass

    def _init_params(self):
        pass

    def __str__(self) -> str:
        out_str = f'this is an empty layer'
        return out_str

    def forward(self, x):
        raise NotImplementedError

    def backward(self, d_last):
        raise NotImplementedError
    
    def update_params(self, l_r):
        pass


class Linear(Layer):
    def __init__(self, in_dim, out_dim) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.params = {
            'weight': None,
            'bias': None
        }

        self.gradient = {
            'd_w': None,
            'd_b': None
        }

        self.cache = {'x': None}

        self._init_params()

    def _init_params(self):
        self.params['weight'] = 0.01 * np.random.randn(self.in_dim, self.out_dim)
        self.params['bias'] = 0.01 * np.zeros((self.out_dim, ))

        self.gradient['d_w'] = np.zeros_like(self.params['weight'])
        self.gradient['d_b'] = np.zeros_like(self.params['bias'])

    def __str__(self) -> str:
        out_str = f'Linear Layer\n\
                    in_dim: {self.in_dim}, out_dim: {self.out_dim}\n\
                    #params: {self.weight.size + self.bias.size}\n'
        return out_str

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, f'Linear input dim should equal to in dim, get {x.shape[-1]}, expect {self.in_dim}'
        self.cache['x'] = x
        return x @ self.params['weight'] + self.params['bias']

    def backward(self, d_last):
        # compute gradient to self params
        # also need to compute gradient for next node
        assert d_last.shape[-1] == self.out_dim, f'Linear d_last dim should equal to out dim, get {d_last.shape[-1]}, expect {self.out_dim}'
        self.gradient['d_w'] = self.cache['x'].T @ d_last
        self.gradient['d_b'] = np.sum(d_last, axis=0)
        dx = d_last @ self.params['weight'].T
        return dx
 
    def update_params(self, l_r):
        self.params['weight'] -= l_r * self.gradient['d_w']
        self.params['bias'] -= l_r * self.gradient['d_b']


class ReLU(Layer):
    def __init__(self) -> None:
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


