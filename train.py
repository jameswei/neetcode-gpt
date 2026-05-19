import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        """
        AdamW is the standard optimizer for transformer
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        """
        `data` is 1D token tensor of integer IDs in shape (vocab_size,).
        """
        print(f"data: {data.shape}")

        """
        training iterations
        for each epoch, sample a random batch, run the forward pass, compute cross-entropy loss with reshaping, backpropogate, and step the optimizer.
        """
        for epoch in range(epochs):
            # set the seed ensures reproducible batch sampling per epoch.
            torch.manual_seed(epoch)

            """
            random sampling from `data` in total of `batch_size`, each sample with `context_length`
            """
            rand_i = torch.randint(high=len(data)-context_length, size=(batch_size,))
            print(f"rand_i: {rand_i.shape}")

            """
            `X` is input tensor in shape (batch_size, context_length, vocab_size)
            `Y` is target tensor shifted `X` right by 1
            """
            X = torch.stack([data[i:i+context_length] for i in rand_i])
            Y = torch.stack([data[i+1:i+1+context_length] for i in rand_i])
            print(f"X: {X.shape}, Y: {Y.shape}")

            """
            forward pass, logits in shape (batch_size, context_length, vocab_size)
            """
            logits = model(X)
            print(f"logits: {logits.shape}")

            """
            reshape `logits` to (batch_size*context_length, vocab_size)
            reshape `Y` to (batch_size*context_length)
            """
            B, T, C = logits.shape
            flatten_logits = torch.reshape(logits, (B * T, C))
            flatten_target = torch.reshape(Y, (B * T,))
            print(f"flatten_logits: {flatten_logits.shape}, flatten_Y: {flatten_target}")

            loss = F.cross_entropy(flatten_logits, flatten_target)
            print(f"loss: {loss.shape}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)
