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
        self.ln1 = nn.Linear(n_layers, n_layers, dtype=torch.float64)
        self.tanh1 = nn.Tanh()
        self.ln2 = nn.Linear(n_layers, n_layers, dtype=torch.float64)
        self.tanh2 = nn.Tanh()
        self.ln3 = nn.Linear(n_layers, 1, dtype=torch.float64)
        self.tanh3 = nn.Tanh()
        # nn.init.xavier_uniform_(self.ln2.weight)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.bias.data.zero_()
            module.weight.data.fill_(0.0)
        # elif isinstance(module, nn.Tanh):
        #     module.weight.data.fill_(1.0)


    def forward(self, x):
        x = self.tanh1(self.ln1(x)) + x
        x = self.tanh2(self.ln2(x)) + x
        x = self.tanh3(self.ln3(x))
        return x

    
    def predict(self, x):
        return np.asscalar(self.forward(x))


        
        
