import numpy as np
from typing import Tuple, List

"""
Batch normalization flips the axis, it normalizes across the batch for each feature.
Different with Layer normalization, which normalizes across features with each sample (axis=1), Batch normalization works on batch for each feature (axix=0).

In training mode, for a given input with shape `(batch_size, features)`:
```
μ_B = mean(x_i)
σ_B^2 = mean((x_i-μ_B)^2)
x^ = (x-μ_B)/sqrt(σ_B^2+ϵ)
y = γ * x^ + β 
```
then need to update statistics for later inference:
```
running_mean = (1-m)*running_mean + m*μ_B
running_var = (1-m)*running_var + m*σ_B^2
```
in inference mode:
```
x^ = (x - running_mean)/sqrt(running_var + ϵ)
"""
class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        """
        `training` indicates if in training mode or inference mode
        """
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        beta = np.array(beta, dtype=np.float64)
        print(f"x: {x}:{x.shape}, gamma: {gamma}:{gamma.shape}, beta: {beta}:{beta.shape}")

        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)
        print(f"running_mean: {running_mean}:{running_mean.shape}, running_var: {running_var}:{running_var.shape}")

        epsilon = np.float64(1e-5)
        momentum = np.float64(momentum)

        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_hat = (x - mean) / np.sqrt(var + epsilon)

            running_mean = running_mean * (1.0 - momentum) + momentum * mean     
            running_var = running_var * (1.0 - momentum) + momentum * var

        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + epsilon)
        
        y = x_hat * gamma + beta
        return (
            np.round(y, 4).tolist(), 
            np.round(running_mean, 4).tolist(), 
            np.round(running_var, 4).tolist()
            )
