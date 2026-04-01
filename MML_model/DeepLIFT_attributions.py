import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import DeepLift
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt

MODEL_ROOT = './models'

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#adapted code from DEEPLIFT tutorial: https://medium.com/@pysquad/deeplift-explained-python-techniques-for-ai-transparency-93d7ca64832d

model = torch.load(f'{MODEL_ROOT}/TPM_models/A2M_TPM_model.pth', weights_only = False)
model.eval()  # Set the model to evaluation mode
model.to(device)
print(model)

inputs = torch.tensor([[0.5, 1.0, 0.0 , 1.0, 2.0, 3.0, 5.0, 1.0, 0.5, 0.85, 0.1, 0.7, 0.6]], dtype=torch.float32).to(device)
baseline = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device)


deep_lift = DeepLift(model)
attributions = deep_lift.attribute(inputs, baselines=baseline)

print(attributions)

features = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
attr = attributions.detach().cpu().numpy()[0]
plt.bar(features, attr, color='skyblue')
plt.title("DeepLIFT Attributions")
plt.ylabel("Attribution Score")
plt.xlabel("Input Features")
plt.show()