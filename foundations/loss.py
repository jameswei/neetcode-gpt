import numpy as np
from numpy.typing import NDArray


class Solution:
    """
    cross-entropy loss measures how far the predicted probabilities are from the `true` labels.
    it's the standard loss function for classification scenarios.
    typically, sigmod and softmax produce the predicted probabilities that cross-entropy evaluates.
    """

    """
    binary cross-entropy handles 2-class problem.
    for example:
        y_true = [1, 0, 1] and y_pred = [0.9, 0.1, 0.8]
    """
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        # a small epsilon
        epsilon = np.float64(1e-7)
        # thanks to NumPy's broadcasting, no need to use loops, simply use standard addition operator `+`
        np.add(y_pred, epsilon)

        bce_loss = -1 * np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.round(bce_loss, 4)

    """
    categorical cross-entropy supports multiple classes problem with one-hot labels.
    for example:
        y_true = [[1,0,0], [0,1,0]] and y_pred = [[0.7,0.2,0.1], [0.1,0.8,0.1]]
    """
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        epsilon = np.float64(1e-7)
        np.add(y_pred, epsilon)

        """
        In NumPy, the axis parameter defines which dimension is being collapsed during an operation.
        axis=0 (Vertical), axis=1 (Horizontal)
        np.sum(matrix, axis=1), NumPy looks at each row and combines everything horizontally into a single value.
        """
        cce_loss = -1 * np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return np.round(cce_loss, 4)
