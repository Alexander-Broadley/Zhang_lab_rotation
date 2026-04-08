import torch
from torch import nn
from torchmetrics.regression import PearsonCorrCoef

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        # Flatten the tensors
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        pearson = PearsonCorrCoef().to(device)
        pearsons = pearson(x, y)

        loss = 1 - pearsons

        # Sum loss across the batch
        return loss