import torch
from torch import nn

#defining simple network with linear projects to MML activation, 
#prediction is linear combination of activation function output
class referenceModel(nn.Module):
    def __init__(self, activation_function, n_tfs):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.leak = 0.01
        self.linear_in = nn.Linear(n_tfs, n_tfs)
        self.linear_out = nn.Linear(n_tfs, 1)

        # activation function
        self.activation_function = activation_function
    
    def forward(self, x):
        #project an input for each TF to activation with a linear layer
        x = self.linear_in(x)
        x = self.activation_function(x)
        x = self.linear_out(x)
        x = x.flatten()
        return(x)