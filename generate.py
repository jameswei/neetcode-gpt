import torch
import torch.nn as nn
from torchtyping import TensorType

"""
Text generation with GPT is an auto-regressive loop: producing text one token at a time by feeding each predication back as input, and repeats.
This is the inference-time procedure using a trained language model as a text generator.

- crop the context to the model's maximum context length.
- feed context through the model to get probabilities at each position.
- only the last position's distribution matters, since it will be used to predict the next token.
- randomly sample a token from the **distribution** using `torch.multinomial`.
- append the sampled token back to the context.
- decodes to token ID to a character.
"""
class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        """
        `context_length` is the length of so called context window.
        if context exceeds the maximum length, eariler tokens are cropped off.
        """
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        res = []
        for i in range(new_chars):
            """
            crop to `context_length` if context exceeds the maximum length.
            """
            print(f"context: {context.shape}")
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            """
            feed through the model,
            produces a tensor in shape (1, T, d)
            """
            logits = model(context)
            print(f"logits: {logits.shape}")

            """
            extract the last position, get a tensor in shape (1, d)
            then apply it to `softmax`, shape won't be changed.
            """
            last_logit = logits[:, -1, :]
            probs = nn.functional.softmax(last_logit, dim=-1)
            print(f"last_logit: {last_logit.shape}, probs: {probs.shape}")

            """
            sample next token in shape (1, 1)
            """
            next_token = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)
            print(f"next_token: {next_token}:{next_token.shape}, item: {next_token.item()}:{type(next_token.item())}")

            """
            append new token back to context and decode
            """
            context = torch.cat((context, next_token), dim=-1)
            print(f"context: {context.shape}")
            res.append(int_to_char[next_token.item()])

        return ''.join(res)
        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
