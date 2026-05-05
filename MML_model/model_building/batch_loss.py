import torch
import math
import numpy as np

#define a function to calculate the MSE 
def batch_loss(dataloader, model, loss_fn):
    #put model in eval mode
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):     
            #create a prediction - this will be 1 value for given sample and target gene
            pred = model(X)

            #calculate prediction loss - this will be one loss value in a distribution across samples per target gene
            loss = loss_fn(pred, y)
            #get loss value
            loss = loss.item()

    return(loss)