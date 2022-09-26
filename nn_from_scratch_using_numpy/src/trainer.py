from src.loss import MSE
from src.MLP import MLP


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


        
        

