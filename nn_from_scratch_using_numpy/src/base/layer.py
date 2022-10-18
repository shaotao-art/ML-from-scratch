class Layer:
    """
    base class for layers
    """
    def __init__(self) -> None:
        self.params = None
        self.gradient = None

    def _init_params(self):
        pass

    def __str__(self) -> str:
        out_str = f'this is an empty layer'
        return out_str

    def forward(self, x):
        raise NotImplementedError

    def backward(self, d_last):
        raise NotImplementedError
    
    