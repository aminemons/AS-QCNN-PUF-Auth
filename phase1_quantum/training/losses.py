"""Training losses for PUF binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUFLoss(nn.Module):
    """
    Binary Cross Entropy + entropy regularizer.

    The entropy term encourages the model to produce confident
    predictions (low entropy), which is desirable for authentication.

    L = BCE(logits, y) + lambda_ent * H(p)
    """

    def __init__(self, lambda_ent: float = 0.01):
        super().__init__()
        self.lambda_ent = lambda_ent
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        # Clamp to avoid log(0)
        probs_c = probs.clamp(1e-7, 1 - 1e-7)
        entropy = -(probs_c * probs_c.log() + (1 - probs_c) * (1 - probs_c).log())
        ent_loss = entropy.mean()

        return bce_loss + self.lambda_ent * ent_loss, bce_loss, ent_loss
