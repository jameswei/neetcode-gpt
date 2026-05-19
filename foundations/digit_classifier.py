import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Handwriting digit classifier is the "hello world" of deep learning.
The architecture is a two-layer MLP.
- Linear(784,512) projects the high-dimensional pixel space into a learned 512-dimensional representation.
- ReLU introduces non-linearity so that the network could learn.
- Dropout(0.2) randomly sets 20% of activations as 0 only during training, 
  this will force the network to spread information across neurons rather than relying on few ones.
- Linear(512,10) projects to 10-class scores, one per digit.
- Sigmoid squashes each score to [0...1]
"""
class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.model = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        output = self.model(images)
        return torch.round(output, decimals=4)
