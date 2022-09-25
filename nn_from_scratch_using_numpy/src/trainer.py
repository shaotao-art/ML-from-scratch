from src.loss import MSE
from src.MLP import MLP

"""
参数的更新到底应该由谁管理:
1. 是直接在每个layer中写一个update_params函数
2. 还是直接将所有的参数copy到trainer中统一管理
3. 如果换一个更加复杂的optimizer上述两种方法中哪个更好

answer to 3:
无论使用何种优化器,最终都是直接把要更新的步子传过去即可,这个步子可能直接是梯度,也可能是加过momentum之后的梯度(如果是这种情况的化,综合momentum的步骤应该在优化器中完成,
而不是在model的layer之中).

综合, 应该在model的layer中实现一个update_params的函数,传入的参数是learning_rate和最终要更新的dW
所以optimizer应该有获得model中所有参数的能力,以便进行momentum的计算。
"""

class Trainer:
    def __init__(self, model:MLP, loss_fn:MSE, learning_rate:float) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.l_r = learning_rate
    
    def forward(self, x, y):
        x = self.model.forward(x)
        loss = self.loss_fn.forward(x, y)
        return loss

    def update_params(self):
        loss_back = self.loss_fn.backward()
        self.model.backward(loss_back)
        self.model.update_params(self.l_r)


        
        

