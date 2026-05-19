import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        safe_shift = z - np.max(z)
        shifted = np.exp(safe_shift)
        return np.round(shifted / np.sum(shifted), 4)