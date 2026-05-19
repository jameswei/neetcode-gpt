import numpy as np
from typing import List


class Solution:
    """
    MLP is a foundation feedforward artificial neural network consisting of at least 3 layers - input, hidden, and output.
    It uses backpropagation and non-linear activation functions to train. 
    """
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        """
        since given parameter `x` is a 1-D matrix(array), so it's a single sample scenario.
        `x` is a row vector with shape 1*D, `W1` with shape D*H, `b1` with shape 1*H
        `z1 = xW1 + b1`, `z1` with shape 1*H
        ReLU as activation func, `a1 = max(0, z1)`, with shape 1*H
        `z2 = a1W2 + b2`, `W2` with shape H*1, `b2` is scalar, `z2` is also scalar
        `y^ = z2`, where `y^` is the prediction, also scalar
        `loss = 1/2 * (y^ - y)^2`, `loss` is scalar
        """
        # need to wrap list as ndarray
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        """
        check each ndarray's shape to guarantee correct vector computing
        x.shape: (2,) == (1,2)
        W1.shape: (2,2), b1.shape: (2,) == (1,2)
        a1.shape: (1,2)
        W2.shape: (1,2), b2.shape: (1,) == (1,1) == scalar
        y_ture.shape: (1,) == (1,1) == scalar
        """
        """
        根据 x 的形状和 W1 的形状来判断“是否需要转置”：
        目标是使 inner dim 相同: “x 的最后一个维度 D 必须等于 W1 的第一个维度”。
        - 如果 x 的形状是 (1, D) 且你期望 z1 的形状是 (1, H)，并且 W1 的形状是 (H, D)，那么需要对 W1 转置，z1 = x @ W1.T
        - 如果 W1 的形状已经是 (D, H)，那么无需转置，z1 = x @ W1
        """

        res = {}
        """
        run forward pass to get all intermediate values, z1, a1, z2
        """
        # x.shape: (1,D)
        # W1.shape: (H,D), need a transpose
        # b1.shape: (1,H)
        # z1.shape: (1,H)
        z1 = x @ W1.T + b1
        # a1.shape: (1,H)
        a1 = np.maximum(0, z1)
        # W2.shape: (D,H), need a transpose to align with a1
        # b2.shape: (1,D), a scalar
        # z2.shape: (1,D), a scalar
        z2 = a1 @ W2.T + b2
        # L.shape: a scalar
        L = np.mean(np.square(z2 - y_true))
        res['loss'] = np.round(L, 4)

        """
        backward propagation
        """
        n = len(y_true) if y_true.ndim > 0 else 1
        dL_dz2 = 2 * (z2 - y_true) / n
        dL_dW2 = dL_dz2.reshape(-1, 1) @ a1.reshape(1, -1)
        dL_db2 = dL_dz2
        res['dW2'] = np.round(dL_dW2, 4).tolist()
        res['db2'] = np.round(dL_db2, 4).tolist()

        dL_da1 = dL_dz2.reshape(1, -1) @ W2
        dL_da1 = dL_da1.flatten()
        dL_dz1 = dL_da1 * (z1 > 0).astype(float)
        dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
        dL_db1 = dL_dz1
        res['dW1'] = np.round(dL_dW1, 4).tolist()
        res['db1'] = np.round(dL_db1, 4).tolist()

        
        return res