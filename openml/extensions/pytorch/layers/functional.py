import torch.nn


class Functional(torch.nn.Module):
    def __init__(self, function, *args, **kwargs):
        super(Functional, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.function(x, *self.args, **self.kwargs)
