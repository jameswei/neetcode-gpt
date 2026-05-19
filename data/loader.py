import torch
from torchtyping import TensorType
from typing import Tuple

"""
Purpose of creating data batches is for training next-token prediction, where `X` is the input and `Y` is the target, 
given a sequence of tokens, model learns to predict what comes next at each position.

- For given `X`, `Y` is shifted right by 1, this is how generate from input data.
- Random sampling is required for each batch creation, that's to say pick a random starting index in the data tensor using `torch.randint`.
  This ensures the model sees different parts of the text in each epoch.
- Finally stack multiple `(X,Y)` into a batch in shape (batch_size, seq_length)

This data loading pattern is widely used by all modern LLMs.
"""
class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        """
        `data` is an encoded text in 1D tensor
        """
        print(f"data: {data}:{data.shape}, batch_size: {batch_size}, seq_length: {context_length}")
        torch.manual_seed(0)
        """
        for given `context_length`, starting index can be [0, N-C]
        but in order to get a shifted right by 1, starting index should be [0, N-C)
        """
        # rand_i = torch.randint(high=len(data)-context_length+1, size=(batch_size,))
        rand_i = torch.randint(high=len(data)-context_length, size=(batch_size,))

        X = torch.stack([data[i:i+context_length] for i in rand_i])
        Y = torch.stack([data[i+1:i+1+context_length] for i in rand_i])

        return (X, Y)