import torch
from torch import nn
from torchmetrics.regression import PearsonCorrCoef

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        #stack into rows for use with torch.corrcoef
        pred_acc_stack = torch.stack([x, y], dim=0)

        #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        #pearson = PearsonCorrCoef().to(device)
        #pearsons = pearson(x, y)

        pearsons = torch.corrcoef(pred_acc_stack)
        print(pearsons)
        loss = 1 - pearsons

        return loss