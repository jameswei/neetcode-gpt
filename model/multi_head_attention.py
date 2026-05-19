import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Multi-headed attention mechanism runs in parallel to focus on multiple types of token relationships.
For example, one head handle subjects to verbs, and another may handle adj before a noun.
Each head operates on a slice of size `attention_dim / num_heads`, and finally their outputs are concatenated together and passed through a linear projection.
"""
class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        each_head_size = attention_dim // num_heads
        self.multi_headed = nn.ModuleList([self.SingleHeadAttention(embedding_dim, each_head_size) for _ in range(num_heads)])
        self.concat_layer = nn.Linear(in_features=attention_dim, out_features=attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """
        input embedding tensor in shape (batch_size, seq_len, embedding_dim)
        produces a tensor in shape (batch_size, seq_len, attention_dim)
        """
        print(f"embedded: {embedded}:{embedded.shape}")

        """
        `nn.ModuleList` acts as an iterable
        """
        res = []
        for m in self.multi_headed:
            res.append(m(embedded))
        
        concat = torch.concat(res, dim=2)
        output = self.concat_layer(concat)

        return torch.round(output, decimals=4)
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
