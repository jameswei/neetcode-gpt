import numpy as np
from numpy.typing import NDArray

"""
The layer normalization fixes huge or tiny numbers by re-centering and re-scaling each layer's output, so that the values can stay in a stable range.
It normalizes each sample independently across its features. For each feature vector `x`:
```
x_i^ = (x_i-μ) / sqrt(σ^2+ϵ) * γ_i + β
μ = mean(x_i)
σ^2 = mean((x_i-μ)^2)
γ and β are learnable scale/shift parameters
ϵ = 10^-5 prevents division by 0
```

"""
class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)

        mean = np.mean(x)
        var = np.mean((x-mean)**2)
        # eps = 10 ** -5
        # better performace without any calculation
        eps = np.float64(1e-5)
        normalized = (x-mean)/np.sqrt(var+eps) 
        out = normalized * gamma.T + beta

        return np.round(out, 5)
