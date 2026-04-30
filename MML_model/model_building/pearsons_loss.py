import torch
from torch import nn
from torchmetrics.regression import PearsonCorrCoef

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):

        #flatten the tensors        
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        #make sure tensors and the Pearsons CorrCoef object are on the same device
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        pearson = PearsonCorrCoef().to(device)

        #calculate pearsons
        pearsons = pearson(x, y)

        #loss function is being minimised, take inverse of pearsons as want to maximise the correlation
        loss = 1 - pearsons

        return loss