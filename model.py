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
        self.layer1 = nn.Tanh()
        self.layer2 = nn.Tanh()

    def forward(self, x):
        # change the structure of this
        # x = nn.ReLU(self.layer1(x.double()))
        # x = nn.ReLU(self.layer2(x.double()))
        # return nn.Softmax(self.layer3(x), dim=-1)
        print(self.weights, x)
        print(from_numpy(np.dot(self.weights, x) + self.biases))
        x = self.layer1(from_numpy(np.dot(self.weights, x) + self.biases))
        print(self.weights)
        print(x)
        return x
    
    def predict(self, x):
        print("x before forward", x)
        return self.forward(from_numpy(x))

    def update(self, values, reward):
        # update weights 
        TD_error = sum(values-reward)
        self.weights += self.lr * TD_error

        
        
