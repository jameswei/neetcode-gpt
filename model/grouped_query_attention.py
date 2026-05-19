import torch
import torch.nn as nn
from torchtyping import TensorType

"""
In MHA, every head has its own K and V projections. however it's expensive to store K and V independently for each head.
MQA uses a single shared K and V for all queries, consuming least memory with a slight drop on quality.
GQA shares K and V across a group of heads, achieving the same quality but with less memory.
"""
class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        print(f"x: {x.shape}")
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")

        """
        With `h` query heads and `g` KV heads, each KV head serves `h/g` query heads. 
        The key operation is `repeat_interleave`: it expands the `g` KV heads to `h` by repeating each one `h/g` times, 
        making the shapes match for standard attention math. 
        """
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        repeat_times = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_times, dim=1)
        v = v.repeat_interleave(repeat_times, dim=1)
        print(f"k: {k.shape}, v:{v.shape}")

        # 4. Compute scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = nn.functional.softmax(scores, dim=-1)

        # 5. Concatenate heads and apply output projection
        scores = (weights @ v).transpose(1, 2).reshape(B, T, -1)
        output = self.output_proj(scores)

        # 6. Return rounded output (decimals=4)
        return torch.round(output, decimals=4)