
class Optim:
    def __init__(self, learning_rate) -> None:
        self.l_r = learning_rate
        self.w_go = None

    def update_w_go(self, i, j, v):
        """
        i: layer i in model's layers
        j: j th element in layer i 's params
        v: value that cover .data and .gradient
        """
        raise NotImplementedError

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
                    self.update_w_go(i, j, v)
                    # update params
                    layer.params[k].data -= self.l_r * self.w_go[i][j]
