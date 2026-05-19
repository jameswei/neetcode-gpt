from typing import List, Dict

"""
- find the longest substring exists in the vocabulary.
- consume that token, then repeat from where you left off.
- if no match found, consume a single character.
"""
class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        tokenized = []
        
        for num in numbers:
            text = str(num)
            tokens = []
            i = 0
            while i < len(text):
                j = i
                longest_end = i + 1
                while j < len(text):
                    if text[i:j+1] in vocab:
                        longest_end = j + 1
                    j += 1
                tokens.append(text[i:longest_end])
                i = longest_end
            tokenized.append(tokens.copy())

        return tokenized

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        total_tokens = 0

        n, i = len(text), 0
        while i < n:
            j = i
            longest_end = i + 1
            while j < n:
                if text[i:j+1] in vocab:
                    longest_end = j + 1
                j += 1
            total_tokens += 1
            i = longest_end
        
        return total_tokens

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        total_tokens = self.count_tokens(text, vocab)
        total_words = len(text.split())

        return round(total_tokens/total_words, 4)