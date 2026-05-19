import torch
import torch.nn as nn
from typing import List, Dict

"""
A trained model may be not a well-trained model. training diagnostics is used to figure out why a model wasn't learnt well:
- activation inspection
- gradients inspection
- loss curve inspection

a systematic debugging process for neural networks:
- look at the loss curve, check learning-rate and gradients if loss curve going up.
- check activations' mean and std_var to see if collapsed to 0 or exploded.
- check gradients to see if vanished or exploded.
- look for dead neurons
"""
class Solution:
    """
    mean, std_var, dead_fraction of output activations
    """
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats: list[dict[str, float]] = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std_var = round(x.std().item(), 4)
                    if x.dim() < 2:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).all(dim=0).float().mean().item(), 4)
                    stats.append({'mean': mean, 'std': std_var, 'dead_fraction': dead_frac})
        return stats

    """
    mean, std_var, norm of each layer's weights gradient
    """
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats: list[dict[str, float]] = []
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean = round(grad.mean().item(), 4)
                std_var = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean, "std": std_var, 'norm': norm})
        
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for act_s in activation_stats:
            if act_s['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for grad_s in gradient_stats:
            if grad_s['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        
        for act_s in activation_stats:
            if act_s['std'] < 0.1:
                return 'vanishing_gradients'
            if act_s['std'] > 10.0:
                return 'exploding_gradients'
        
        return 'healthy'

