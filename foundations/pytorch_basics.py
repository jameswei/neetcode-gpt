import torch
import torch.nn
from torchtyping import TensorType

"""
A tensor is the fundamental data structure in PyTorch, essentially a multi-dimensional array, similar with `NDArray` in NumPy.
It can run on a GPU for hardware-accelerated math computations. In a real neural network, every weight, input, and gradient is a tensor.

Tensor's rank indicates the number of dimensions, starts from 0. Rank 0 is a single number, rank 1 is a 1-D vector, rank 2 is a 2-D matrix, 
and rank 3+ represents stacks of matrices.
The following 4 operations are most common used in every model.
"""
# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    """
    `reshape()` changes given torsor's dimension
    """
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # columns is given, rows can be auto-derived
        # resh = torch.reshape(to_reshape, (-1, 2))
        # or explicitly computed
        M, N = to_reshape.shape
        resh = torch.reshape(to_reshape, ((M*N)//2, 2))
        return torch.round(resh, decimals=4)

    """
    for a given dimension by `dim=`, operation will perform along the dimension's direction, finally that dimension get collapsed (disappeared).
    """
    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        avg = torch.mean(to_avg, dim=0)
        return torch.round(avg, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        concat = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(concat, decimals=4)

    """
    mse loss = mean((pred-target)^2)
    """
    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        mse_loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(mse_loss, decimals=4)
