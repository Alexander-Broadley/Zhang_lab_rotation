#early stopping object adapted from: https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.previous_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.previous_loss is None:
            self.previous_loss = val_loss
        elif abs(self.previous_loss - val_loss) >= self.delta:
            self.no_improvement_count = 0
            self.previous_loss = val_loss
        else:
            self.previous_loss = val_loss
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")