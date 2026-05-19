import numpy as np
from numpy.typing import NDArray
from typing import Tuple

"""
training loop for a single neuron linear regression
"""
class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        print(f"===input===\nX: {X}:{X.shape}, \ny: {y}:{y.shape}")
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        """
        initialize weights to 0 and bias to 0.
        """
        sample_n, feat_n = X.shape
        # return an array with given shape, filled with 0.0
        weights = np.zeros(feat_n, dtype=np.float64)
        print(f"weights: {weights}:{weights.shape}")
        bias = np.float64(0.0)
        lr = np.float64(lr)

        for _ in range(epochs):
            """
            feed forward
            """
            # compute prediction
            y_hat = X @ weights + bias

            # compute mse as loss
            error = y_hat - y
            loss = np.mean(np.square(error))

            """
            backward propagation
            """
            # compute vectorized gradients
            dw = (2.0 / sample_n) * (error @ X)
            db = (2.0 / sample_n) * np.sum(error)

            # update weights
            weights = weights - lr * dw
            bias = bias - lr * db

        return (np.round(weights, 5), np.round(bias, 5))