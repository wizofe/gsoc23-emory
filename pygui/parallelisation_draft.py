import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Assume we are running on a machine with multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If there's more than one GPU
if torch.cuda.device_count() > 1:
  print("We have ", torch.cuda.device_count(), " GPUs!")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Initialize the model
model = SimpleModel()

# If there are multiple GPUs, wrap the model with nn.DataParallel
# If not, the model stays as is
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

# Now the model will run on multiple GPUs and the forward pass should be faster

