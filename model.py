import numpy as np
import torch
from torch import nn
from torch import from_numpy

class NeuralNetwork(nn.Module):
    def __init__(self, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.lr = lr
        # self.weights = np.full(n_layers, 0.001)
        # self.biases = np.zeros(n_layers)
        # self.ln1 = nn.Linear(n_layers, n_layers, dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.ln2 = nn.Linear(n_layers, 1, dtype=torch.float64)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # elif isinstance(module, nn.Tanh):
        #     module.weight.data.fill_(1.0)


    def forward(self, x):
        x = self.ln2(x)
        x = self.tanh(x)
        # x = self.ln2(x)
        return x

    
    def predict(self, x):
        return np.asscalar(self.forward(x))


        
        
