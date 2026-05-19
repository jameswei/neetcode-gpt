import numpy as np
from numpy.typing import NDArray

"""
Positional encoding is original from 'Attention Is All You Need' paper.
Its purpose is injecting sequence ordering information, like 'this is the ith word in the sentence', into a Transformer. 
Because the core self-attention mechanism processes all tokens simutenously ranther than sequentially. 
It inherently cannot distinguish between "dog bites man" and "man bites dog".
"""
class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        """
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Use np.arange() to create position and dimension index vectors,
        then compute all values at once with broadcasting (no loops needed).
        Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        """

        """
        create a `PE` vector with the same shape (seq_len, d_model) with word embedding vector and filled with 0s.
        before input into transformer, `PE` will be added to input like `x += PE[:seq_len]`.
        """
        PE = np.zeros((seq_len, d_model))
        print(f"PE: {PE}:{PE.shape}")

        """
        `pos` is the position of word in the sequence, reshape to a vector with shape (seq_len, 1)
        """
        pos = np.arange(seq_len).reshape(-1, 1)
        print(f"pos: {pos}:{pos.shape}")

        """
        `i` goes from 0 to d_model/2-1, `2i` means even indices, `2i+1` means odd indices, cover all indices of d_model
        `_2i` is a vector with shape (d_model/2,)
        `div_term` is derived from: a^b == e^(b*ln(a))
        10000^(2i/d_model) == e^(2i/d_model * ln(10000))
        `-1*` is convert from division to multiplication
        """
        _2i = np.arange(0, d_model, 2)
        div_term = np.exp(-1*_2i*np.log(10000)/d_model)

        """
        `pos*div_term` produces vector with shape (seq_len, d_model/2)
        it is an element-wise multiplication by numpy broadcasting

        PE[:, start:stop:step] is the standard slice
        first part `:` means all rows on dimension=0
        second part `0::2` and `1::2` mean on dimension=1 only [0,2,4...] rows and [1,3,5...] rows
        """
        PE[:, 0::2] = np.sin(pos * div_term)
        PE[:, 1::2] = np.cos(pos * div_term)

        return np.round(PE, decimals=5)