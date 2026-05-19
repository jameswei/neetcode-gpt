from typing import List

"""
Byte Pair Encoding (BPE) tokenization is widely used in most modern LLMs.
1. Split corpus into a list of individual characters.
2. For each merge step:
   a. Count frequency of all adjacent token pairs
   b. Find the most frequent pair (break ties lexicographically)
   c. Merge all non-overlapping occurrences left to right
   d. Record the merge as [token_a, token_b]
3. Return the list of merges performed.

After BPE training, a given new text stirng, apply the learned merges, which deterministically converts any text into a sequence of subword tokens.
In GPT-4, there're around 100,000 tokens in its BPE vocabulary
"""
class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        if len(tokens) == 1:
            return []

        merges = []
        for _ in range(num_merges):
            if len(tokens) == 1:
                break
            
            pair_count = {}
            # count frequency of all adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_count[pair] = pair_count.get(pair, 0) + 1
        
            if len(pair_count) == 0:
                break
            
            # find the most frequent pair
            most_freq = max(pair_count.values())
            candidate_pairs = sorted(p for p, c in pair_count.items() if c == most_freq)
            best_pair = candidate_pairs[0]
            merges.append([best_pair[0], best_pair[1]])

            # merge all non-overlapping occurrences
            new_tokens = []
            i = 0
            while i < len(tokens)-1:
                if tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0]+best_pair[1])
                    # skip by 2 if merged
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return merges