import torch
import torch.nn as nn
from typing import Tuple, Optional

"""
During auto-regressive generation, **each new token runs a full forward pass**, 
in each attention layer, the model computes K, Q, V from the entire context, for new token, the total work is O(N^2).

The key insight is that:
- K and V for previous tokens don't change between generation steps. 
- Only computes new K and V for new token, then update to cache.
- Only Q needs to be computed for new token.

This optimization reduces computation from O(N^2) to O(N).
"""
class KVCache:
    def __init__(self):
        """
        both in shape (batch_size, context_length, model_dim)
        """
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the corresponding cache along the sequence/context dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None or self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            """
            append to cache along a specific dimension is tensor concatenation using `torch.cat`
            """
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        
        return (self.cache_k, self.cache_v)

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. If kv_cache is None, create a new KVCache
        if kv_cache is None:
            kv_cache = KVCache()
        
        # 3. Update the cache with the new K and V
        full_k, full_v = kv_cache.update(k, v)

        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        scores = q @ torch.transpose(full_k, -2, -1) / (full_k.shape[-1]**0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ full_v  

        # 5. Return (rounded output, kv_cache)
        return torch.round(output, decimals=4), kv_cache
