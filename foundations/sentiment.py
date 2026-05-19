import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Sentiment analysis is the foundation of traditional NLP, like Transform/Self-attention to modern LLMs.
A bag of words classifier is composed of:
- an embedding layer maps each token ID to a learnt 16-dimensional vector.
- averaging across the sequence collapses variable-length input into a fixed-size representation.
- linear layer + sigmoid output a probability between [0...1]
"""
class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> mean(dim=1) -> Linear(16, 1) -> Sigmoid
        """
        This "average-then-classify" pattern is the standard "Bag of Words" model.
        """
        self.model = nn.Sequential(
            nn.Embedding(vocabulary_size, 16),
            Mean(dim=1),
            nn.Linear(in_features=16, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        # Return a B, 1 tensor and round to 4 decimal places
        torch.manual_seed(0)
        output = self.model(x)
        return torch.round(output, decimals=4)


class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: TensorType[float]):
        return x.mean(dim=self.dim)
