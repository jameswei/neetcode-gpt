import numpy as np
from numpy.typing import NDArray

"""
Neural network cannot process raw text, like digit classification, it need numbers.
Word embedding map each 'token' to a vector where 'similar' words (in semantic) sit close together.
It's the foundation for language understanding, instead of representing words as one-hot vectors, which are sparse and high-dimensional,
but a dense and low-dimensional vector contains semantic meaning. 



An embedding table is simply a matrix with shape (vocab_size, embed_dim), query the embedding for a given token_id: embedding[token_id]
"""
class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        return np.round(embeddings[token_ids], 5)
