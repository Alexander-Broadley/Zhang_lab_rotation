#define a function for training a single epoch

#define the training loop
def train_one_epoch(dataloader, model, loss_fn, optimiser):
    #losses = []
    #size = len(dataloader.dataset)
    
    #put model in train mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):     
        #zero the gradient for each batch
        optimiser.zero_grad()

        #create a prediction
        pred = model(X)
        
        #calculate prediction loss
        loss = loss_fn(pred, y)

        #backpropogate for given loss
        loss.backward()
        optimiser.step()

        loss = loss.item()
        #losses.append(loss)
        return(loss)
