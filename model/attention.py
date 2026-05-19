import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Self-attention is the most crutial part makes transformer work.

Neural network receives sentences as sequences of numbers, but not every token carries same weight. Model needs a way to figure out which tokens deserve more focus.
`forward` method accepts the embedding vector for every token in the sequence, then produces an output vector with size `attention_dim` per token.
These output vectors capture the contextually and semantically relevant information.

Through training, the model discovers which token pairs carry meaningful relationships, which gives transformer deep understanding of language structure.

Each token's embedding is projected into 3 vectors: Q(query), K(key), V(value) using learned weight matrices.
"""
class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        """
        define 3 linear projections (Key, Query, Value) with `bias=False`
        """
        self.key_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.query_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.value_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # torch.manual_seed(0)
        """
        project embedded vector into Key, Query, and Value layers.
        """
        k = self.key_layer(embedded)
        q = self.query_layer(embedded)
        v = self.value_layer(embedded)

        score = q @ torch.transpose(k, 1, 2)
        context_length, attension_dim = k.shape[1], k.shape[2]
        score = score / attension_dim**0.5
        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        score = score.masked_fill(mask, float('-inf'))
        score = nn.functional.softmax(score, dim=2)

        return torch.round(score @ v, decimals=4)
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places