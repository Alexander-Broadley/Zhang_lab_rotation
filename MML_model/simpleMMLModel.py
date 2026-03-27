import torch
from torch import nn

#defining simple network with linear projects to MML activation, 
#prediction is linear combination of activation function output
class SimpleMMLModel(nn.Module):
    def __init__(self, activation_function, n_tfs):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.leak = 0.01
        self.linear_in = nn.Linear(n_tfs, n_tfs)
        self.linear_out = nn.Linear(n_tfs, 1)

        # activation function
        self.activation = activation_function['activation']
        self.delta = activation_function['delta']
        self.onestepdelta_activation_factor = activation_function['onestepdelta']
    
    def forward(self, x):
        #project an input for each TF to activation with a linear layer
        expressions = self.linear_in(x)
        expressions = self.activation(expressions, self.leak)
        expressions = self.linear_out(x)
        expressions = expressions.flatten()
        return(expressions)