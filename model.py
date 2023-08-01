import numpy as np
from torch import nn
from torch import from_numpy

class NeuralNetwork(nn.Module):
    def __init__(self, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.lr = lr
        self.weights = np.full(n_layers, 0.001)
        self.biases = np.zeros(n_layers)
        self.layer1 = nn.Linear(n_layers, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_layers)

    def forward(self, x):
        # change the structure of this
        x = nn.ReLU(self.layer1(x.double()))
        x = nn.ReLU(self.layer2(x.double()))
        return nn.Softmax(self.layer3(x), dim=-1)
    
    def predict(self, x):
        return self.forward(from_numpy(x))

    def update(self, values, reward):
        # update weights 
        TD_error = sum(values-reward)
        self.weights += self.lr * TD_error

        
        
