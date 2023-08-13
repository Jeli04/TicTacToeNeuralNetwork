import numpy as np
from torch import nn
from torch import from_numpy

class NeuralNetwork(nn.Module):
    def __init__(self, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.lr = lr
        # self.weights = np.full(n_layers, 0.001)
        # self.biases = np.zeros(n_layers)
        self.ln1 = nn.Linear(n_layers, n_layers)
        self.tanh = nn.Tanh()
        self.ln2 = nn.Linear(n_layers, 1)

    def forward(self, x):
        x = self.ln1(x)
        x = self.tanh(x)
        x = self.ln2(x)
        return x

        # print(self.weights, x)
        # print(from_numpy(np.dot(self.weights, x) + self.biases))
        # x = self.layer1(from_numpy(np.dot(self.weights, x) + self.biases))
        # print(self.weights)
        # print(x)
        # return x
    
    def predict(self, x):
        print("x before forward", x)
        return self.forward(from_numpy(x))


        
        
