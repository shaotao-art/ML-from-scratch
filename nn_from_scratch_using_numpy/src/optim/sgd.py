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
                layer_lst = []
                if layer.params is not None:
                    for j, (k, v) in enumerate(layer.params.items()):
                        layer_lst += [v.gradient]
                        # 此处应为 np.zeros
                    self.w_go.append(layer_lst)
                else:
                    self.w_go.append(layer_lst)


        for i, layer in enumerate(model.layers):
            if layer.params is not None:
                for j, (k, v) in enumerate(layer.params.items()):
                    # momentum update
                    self.w_go[i][j] = self.momentum * self.w_go[i][j] + (1 - self.momentum) * v.gradient
                    # update params
                    layer.params[k].data -= self.l_r * self.w_go[i][j]

                

        

