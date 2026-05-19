import torch
from typing import List, Tuple

"""
This step is prior to Data loader, which consumes pre-tokenized integers, whereas Batch loader is fed into raw text.
"""
class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        words = raw_dataset.split()
        rand_i = torch.randint(high=len(words)-context_length, size=(batch_size,))

        X = [words[i:i+context_length] for i in rand_i]
        Y = [words[i+1:i+1+context_length] for i in rand_i]
        return (X, Y)