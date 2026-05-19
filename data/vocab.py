from re import I
from typing import Dict, List, Tuple

"""
A simpler character-level encoding, which keeps the vocabulary small enough.
"""
class Solution:
    """
    extract all unique characters from the given text, sort them lexicographically, and assign each character a unique integer starting from 0.
    return mapping in dual-direction.
    """
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        s_to_i = {}

        chars = sorted(list(text))
        n = len(chars)
        i, no = 0, 0
        while i < n:
            if chars[i] not in s_to_i:
                s_to_i[chars[i]] = no
                no += 1
            i += 1

        return (s_to_i, {v: k for k, v in s_to_i.items()})

    """
    convert a string into a list of integers using str_to_int mapping.
    """
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded: list[int] = []
        for c in text:
            encoded.append(stoi[c])
        
        return encoded

    """
    convert a list of integers back into a string using int_to_str mapping.
    """
    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded: list[str] = []
        for n in ids:
            decoded.append(itos[n])
        
        return ''.join(decoded)
