"""
Hybrid Classical-Quantum model for PUF binary authentication.

Pipeline:
  [64-bit challenge] → ClassicalEncoder(64→128→64→8)
                     → TorchVQC(8 qubits, 6 layers)
                     → ClassicalHead(8→1)
                     → sigmoid → binary response
"""

import torch
import torch.nn as nn
from .quantum_circuit import TorchVQC


class ClassicalEncoder(nn.Module):
    """Maps 64-bit PUF challenge to 8 qubit rotation angles."""

    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
            nn.Tanh(),   # output in [-1, 1] → scaled to [-pi, pi] by VQC
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.net(x)          # [B, 8]
        return angles * torch.pi      # scale to [-pi, pi]


class ClassicalHead(nn.Module):
    """Maps 8 PauliZ expectation values to a binary prediction."""

    def __init__(self, in_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # [B]


class HybridQCNN(nn.Module):
    """
    Full hybrid Classical-Quantum model.

    Training phases (controlled externally by trainer):
      Phase 1 (warm-up): freeze VQC, train encoder + head
      Phase 2 (joint):   unfreeze all, end-to-end backprop through VQC
    """

    def __init__(self, n_bits: int = 64, n_qubits: int = 8, n_layers: int = 6):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.encoder = ClassicalEncoder(in_features=n_bits, out_features=n_qubits)
        self.vqc = TorchVQC(n_qubits=n_qubits, n_layers=n_layers)
        self.head = ClassicalHead(in_features=n_qubits)

    def freeze_vqc(self):
        for p in self.vqc.parameters():
            p.requires_grad_(False)

    def unfreeze_vqc(self):
        for p in self.vqc.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_bits] → logits [B]"""
        angles = self.encoder(x)
        expectations = self.vqc(angles)
        logits = self.head(expectations)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Returns hard binary predictions {0, 1}."""
        logits = self.forward(x)
        return (logits > threshold).long()

    def count_parameters(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        vqc = sum(p.numel() for p in self.vqc.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {"encoder": enc, "vqc": vqc, "head": head, "total": enc + vqc + head}
