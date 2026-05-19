import torch
import torch.nn as nn
import math
from typing import List

"""
Each layer multiplies input with a weight matrix. If the weights are too large, the output grows exponentially layer by layer. If too small, it shrinks to 0.
After 10 layers the signal is either extremely large or numeriacally dead.

Weight initialization makes each layer preserves the **variance** of input, then activations will stay in a numerically stable range no matter how many layers deep.
```
Var(X) = mean((x_i-x^)^2)
```
variance is used to measure how far each data is away from the mean. This is similar with what we measure prediction and expectation using MSE.
"""
class Solution:

    """
    xavier initialization set the standard deviation, applicable to Sigmoid function
    """
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * xavier_std
        return torch.round(weights, decimals=4).tolist()

    """
    kaiming initiailization
    """
    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        kaiming_std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * kaiming_std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        print(f"num_layers: {num_layers}, input_dim: {input_dim}, hidden_dim: {hidden_dim}")
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        print(f"dims: {dims}")
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i]+dims[i+1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            w = torch.randn(dims[i+1], dims[i]) * std
            weights.append(w)
        print(f"weights: {weights}")
        
        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds