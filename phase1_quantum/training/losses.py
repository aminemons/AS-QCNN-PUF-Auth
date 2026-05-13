"""Training losses for PUF binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUFLoss(nn.Module):
    """
    Hinged Loss + variance regularizer.

    The hinge loss provides a steady gradient across a margin, preventing the
    flatlining seen with BCE in quantum models. The variance penalty ensures
    the model doesn't collapse to outputting 0 for everything.

    L = Hinge(logits, y) + lambda_ent / (Var(logits) + 1e-6)
    """

    def __init__(self, lambda_ent: float = 0.01):
        super().__init__()
        self.lambda_ent = lambda_ent

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Map targets {0, 1} to {-1, 1}
        y_signed = targets * 2.0 - 1.0

        # Hinge Loss: max(0, 1 - y * logit)
        hinge_loss = F.relu(1.0 - y_signed * logits).mean()

        # Variance Penalty: penalize low variance in logits across the batch
        var_penalty = 1.0 / (logits.var() + 1e-6)

        total_loss = hinge_loss + self.lambda_ent * var_penalty
        return total_loss, hinge_loss, var_penalty
