import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Each transformer combines 2 sub-layers with residual connection and layer-normalization in a Pre-Norm architecture.
LayerNorm is applied BEFORE each sub-layer, this differs from the original "Attention is All You Need" approach but trains better and stable.
```
x = x + multi_headed_attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```
The residual connections adds input directly to the output of sub-layer like `x + f(x)`.
FFN is a 2-layer MLP:
```
FFN(x) = Dropout(ReLU(x*W_1 + b_1)* W_2 + b_2)
```
"""
class TransformerBlock(nn.Module):

    """
    `model_dim` is both embedding dimension and attention output.
    """
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.multi_headed_attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
        """
        `VanillaNeuralNetwork` sub-module implements the FFN with 4x expansion, ReLU, and dropout.
        """
        self.feed_forward = self.VanillaNeuralNetwork(model_dim)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)


    """
    produces a tensor in shape (batch_size, seq_len, model_dim)
    """
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        embedded = embedded + self.multi_headed_attention(self.layer_norm_1(embedded))
        embedded = embedded + self.feed_forward(self.layer_norm_2(embedded))

        return torch.round(embedded, decimals=4)
        # Two residual connections with Pre-LN:
        #   x = x + attention(layer_norm_1(x))
        #   x = x + feed_forward(layer_norm_2(x))
        # Return result rounded to 4 decimal places

    class MultiHeadedSelfAttention(nn.Module):

        class SingleHeadAttention(nn.Module):
            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)

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

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.att_heads = nn.ModuleList()
            for i in range(num_heads):
                self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
            self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            head_outputs = []
            for head in self.att_heads:
                head_outputs.append(head(embedded))
            concatenated = torch.cat(head_outputs, dim = 2)
            return self.output_proj(concatenated)

    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
