import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Self-attention is the most crutial part makes transformer work.

Neural network receives sentences as sequences of numbers, but not every token carries same weight. 
Model needs a way to figure out which tokens deserve more focus.
`embedded` is a embedding vector for every token in the sequence, 
then produces an output vector with size `attention_dim` per token.
These output vectors capture the contextually and semantically relevant information.

Through training, the model discovers which token pairs carry meaningful relationships, 
which gives transformer deep understanding of language structure.

Each token's embedding is projected into 3 new embedding vectors: Q(query), K(key), V(value) using trained weights and (biases).
This transformation is done by applying different weights and biases to each token embedding.

Once created, Q compares with K to measure relevance, which is used to weight the V.
"""
class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        print(f"embedding_dim: {embedding_dim}, attention_dim: {attention_dim}")
        super().__init__()
        torch.manual_seed(0)
        """
        each projection is a simple linear layer.
        define 3 linear projections (Key, Query, Value) with `bias=False`
        """
        self.key_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.query_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.value_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """
        for a given embedding vector with shape (batch_size, seq_len, embedding_dim), 
        produces Q/K/V vectors with shape (B, L, attention_dim)
        - axis=0, numbers of sequences
        - axis=1, numbers of tokens in each sequence
        - axis=2, length of vector of each embedded token
        `embed_dim` is the independent super parameter, not depends on tokenizer.
        """
        print(f"embedded: {embedded.shape}")
        # torch.manual_seed(0)
        """
        project embedded vector through Key, Query, and Value layers.
        attention is built from 3 matrix multiplies:
        - scores = Q * K^T
        - norm_scores = softmax(scores)
        - norm_scores * V

        the total time complexity is O(L^2*d)
        """
        K = self.key_layer(embedded)
        Q = self.query_layer(embedded)
        V = self.value_layer(embedded)
        print(f"k: {K.shape}, \nq: {Q.shape}, \nv: {V.shape}")

        """
        Q * K^T computes a similarity score between each pair of tokens.
        Q's shape (B, L, attention_dim), `K.transpose(1,2)`, produces (B, attention_dim, L)
        Q @ K^T will get shape (B, L, L)
        
        This step is q comparing with k to measure relevance.
        Each row represents a position of 'query' and each column represents a position of 'key', 
        higher score means more relevant between them. 
        """
        # O(L*L*d)
        score = Q @ K.transpose(1, 2)
        print(f"score: {score.shape}")

        """
        this step is scaling, dividing by square root if K's `d_model` or `attention_dim`, 
        to prevents the dot product from growing too large and then gradient will tend towards 0.
        """
        _, seq_len, attention_dim = K.shape
        scaled_score = score / attention_dim**0.5

        """
        a causal mask will be applied before `softmax` to ensure each token can only attend to itself and earlier tokens.
        token at [i] only attends to token at [j] where j <= i in the (L, L) vector.
        create a vector in shape (L, L) which is aligned with output from `Q @ K^T` and filled with 1s.
        a lower triangular transformation only keeps original values (1) on bottom-left part, others are replaced with 0.
        for example, seq_len = 4,
        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
        """
        lower_triangular = torch.tril(torch.ones(seq_len, seq_len))
        """
        converts to a boolean vector in same shape, only 0 -> True
        """
        mask = lower_triangular == 0
        """
        applies mask to `scaled_scores`, replaces positions that not allowed to attend with '-inf'.
        """
        masked_score = scaled_score.masked_fill(mask, float('-inf'))
        """
        '-inf's will be ignored in `softmax`
        """
        norm_score = nn.functional.softmax(masked_score, dim=2)

        output = norm_score @ V

        return torch.round(output, decimals=4)
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places