import numpy as np
from numpy.typing import NDArray

class Solution:
    """
    linear regression is the simplest predictive model.
    dot production of feature matrix and weight vector computes the forward pass, which is the prediction value.
    - a feature matrix has `n` rows and each row has `m` columns, each row is one data point with `m` attributes.
    - weights represents model's learnable parameters, stored as a vector with `m` length matching columns in feature matrix.
    eacho prediction is a weighted sum of all features, weighted means each feature multiply with a weight
    """
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        pred = np.dot(X, weights)
        return np.round(pred, 5)

    """
    to evaluate how well the model fits, that's to say, how far the prediction value from the target value.
    mean squared error between predictions and ground truth.
    sum((pred[i] - expt)^2)/N
    squaring the error (difference) does 2 useful things:
    - it makes all errors positive, so errors will not cancel out each other.
    - it penalizes large errors much more than tiny ones, which prioritizes the worst prediction.
    """
    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse = np.mean(np.pow(model_prediction-ground_truth, 2))
        return np.round(mse, 5)
