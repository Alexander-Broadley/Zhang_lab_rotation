import torch
from torch import nn

torch.manual_seed(1475460913)

#edited from claude and stack overflow
class OneToOneLinear(nn.Module):    
    def __init__(self, n):
        super().__init__()
        #same number of weights as inputs (one each), initialised randomly
        self.weight = nn.Parameter(torch.randn(n)) 
        #same number of biases as inputs (one each), initialised as zeros  
        self.bias   = nn.Parameter(torch.randn(n))  

    def forward(self, x):
        #then return one output per input, which is the the expected linear results
        return x * self.weight + self.bias


#defining simple network with linear projects to MML activation, 
#prediction is linear combination of activation function output
class SimpleMMLModel(nn.Module):
    def __init__(self, activation_function, n_tfs):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.leak = 0.01
        self.linear_in = OneToOneLinear(n_tfs)
        self.linear_out = nn.Linear(n_tfs, 1)

        # activation function
        self.activation = activation_function['activation']
        self.delta = activation_function['delta']
        self.onestepdelta_activation_factor = activation_function['onestepdelta']
    
    def forward(self, x):
        #project an input for each TF to activation with a linear layer
        x = self.linear_in(x)
        x = self.activation(x, self.leak)
        x = self.linear_out(x)
        x = x.flatten()
        return(x)