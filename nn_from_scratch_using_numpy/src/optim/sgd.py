import numpy as np
from base.optim import Optim


class SGD(Optim):
    def __init__(self, learning_rate, momentum) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum

    def update_w_go(self, i, j, v):
        self.w_go[i][j] = self.momentum * self.w_go[i][j] + (1 - self.momentum) * v.gradient
        


                

        

