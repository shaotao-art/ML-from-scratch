"""
对每层中的每个参数, 都可能记录其running mean等参数,该职责应该交给 optim 而不是layer.
若要

"""

import numpy as np

class SGD:
    def __init__(self, learning_rate, momentum) -> None:
        self.l_r = learning_rate
        self.momentum = momentum
        self.w_go = None

    def step(self, model):
        # init params
        if self.w_go is None:
            self.w_go = []
            for i, layer in enumerate(model.layers):
                if layer.gradient is not None:
                    # 此处应为 np.zeros
                    self.w_go.append(layer.gradient)
                else:
                    self.w_go.append(None)

        for i, layer in enumerate(model.layers):
            if layer.params is not None:
                for k, v in layer.gradient.items():
                    # momentum update
                    self.w_go[i][k] = self.momentum * self.w_go[i][k] + (1 - self.momentum) * v
                    # update params
                    layer.params[k[2:]] -= self.l_r * self.w_go[i][k]

                

        

