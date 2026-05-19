import numpy as np
from typing import List

"""
RMS normalization is a modern but simpler normalization without mean subtraction and no shift β.

```
rms = sqrt(mean(x^2) + ϵ)
x^ = x / rms
y =  γ * x^
"""
class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        print(f"x: {x}:{x.shape}, gamma: {gamma}:{gamma.shape}")
        epsilon = np.float64(eps)

        rms = np.sqrt(np.mean(np.square(x), axis=0) + epsilon)
        x_hat = x / rms
        y = x_hat * gamma
        return np.round(y, 4)