import torch
import torch.nn as nn
from typing import List

"""
Sometimes neural networks fail silently, model trains, the loss decreases, but maybe 60% neurons are dead.
```
ReLU = max(0, X)
```
A dead ReLU means its output is 0 and gradient is also 0, its weights never update if the pre-activation input is negative for every sample in the batch.
the damage of ReLU death may cascade: a downstream layer loses an input, increasing the change that its neurons die too.
"""
class Solution:

    """
    measures per ReLU layer according to the output for every sample.
    """
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        dead_fracs = []
        with torch.no_grad():
            # iterate through `model.children()` and check if `isinstance(module, nn.ReLU)`
            for module in model.children():
                # forward the input through model layer for layer
                x = module(x)
                if isinstance(module, nn.ReLU):
                    dead_frac = (x == 0).all(dim=0).float().mean().item()
                    dead_fracs.append(round(dead_frac, 4))

        return dead_fracs

    """
    the order reflects the priority
    """
    def suggest_fix(self, dead_fractions: List[float]) -> str:
        """
        1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        2. 'reinitialize' if the first layer has dead fraction > 0.3
        3. 'reduce_learning_rate' if dead fraction strictly increases with depth AND the last layer's fraction > 0.1
        4. 'healthy' if max dead fraction < 0.1
        5. 'healthy' otherwise
        """
        if len(dead_fractions) == 0:
            return "healthy"
        
        max_frac = max(dead_fractions)
        if max_frac < 0.1:
            return "healthy"
        
        if max_frac > 0.5:
            return "use_leaky_relu"
        
        if dead_fractions[0] > 0.3:
            return "reinitialize"
        
        strictly_incr = True 
        last_frac = 0.0
        for i, frac in enumerate(dead_fractions):
            if i == len(dead_fractions)-1 and strictly_incr and frac >= last_frac and frac > 0.1:
                return "reduce_learning_rate"
            
            if strictly_incr:
                strictly_incr = frac >= last_frac
                last_frac = frac

        return "healthy" 
