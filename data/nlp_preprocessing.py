import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

"""
raw text is un-processable to a neural network util it becomes numbers.

There's a 'pre-processing', which converts text strings with variable length into fixed-size numerical tensors through a pipeline:
- tokenization splits text into individual tokens.
- vocabulary construction collects all unique words, sorts them, and assigns each one a unique integer ID starting from 1 (0 is reserved for padding).
- encoding replaces each word with its integer ID, turning sentences into numeric sequences.
- padding fills shorter sequences with 0s on the right to make all sequences in a batch have the same length and are able to stack into the rectangular tensor.
"""
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        """
        combine positive and negative sentences together
        build a sorted mapping words to IDs starting from 1
        encode into a tensor of IDs
        padding if necessary
        """
        combined_sent = positive + negative
        sorted_vacab = sorted({word for sent in combined_sent for word in sent.split()})
        word_to_id = {word: idx+1 for idx, word in enumerate(sorted_vacab)}

        encoded = [torch.tensor([word_to_id[w] for w in sent.split()]) for sent in combined_sent]
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)