"""
Pure PyTorch GPU-native Variational Quantum Circuit.

No cuQuantum / lightning.gpu required.
Implements AngleEmbedding + StronglyEntanglingLayers on CUDA.

State vector dimension: 2^n_qubits (256 for 8 qubits — trivial on GPU).
Batch dimension is fully parallelized on GPU.

Reference architecture from the IEEE paper Section II:
  - AngleEmbedding: Ry(x_i) on qubit i
  - StronglyEntanglingLayers: Rz-Ry-Rz rotations + CNOT ring
  - Measurement: PauliZ expectation values
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TorchVQC(nn.Module):
    """
    Variational Quantum Circuit implemented entirely in PyTorch.
    Runs on CUDA with full batch parallelism.

    Args:
        n_qubits:  number of qubits (default 8 → state dim 256)
        n_layers:  number of strongly-entangling layers (default 6)
        init_scale: weight initialization scale
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 6, init_scale: float = 0.01):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.n_layers = n_layers

        # Trainable: 3 rotation angles (Rz, Ry, Rz) per qubit per layer
        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * init_scale
        )

        # Precompute CNOT indices (ring topology: i -> (i+1) % n)
        self._register_cnot_masks()

    # ------------------------------------------------------------------
    # CNOT mask precomputation
    # ------------------------------------------------------------------
    def _register_cnot_masks(self):
        """Precompute index tensors for CNOT operations (no recompute per batch)."""
        n = self.n_qubits
        for ctrl in range(n):
            tgt = (ctrl + 1) % n
            # Indices where control bit == 1
            indices = torch.arange(2 ** n)
            ctrl_is_1 = (indices >> (n - 1 - ctrl)) & 1 == 1
            # Bit-flip mask for target
            flip_mask = 1 << (n - 1 - tgt)
            flipped = indices ^ flip_mask
            self.register_buffer(f"cnot_{ctrl}_{tgt}_src", indices[ctrl_is_1])
            self.register_buffer(f"cnot_{ctrl}_{tgt}_dst", flipped[ctrl_is_1])

    # ------------------------------------------------------------------
    # Gate matrices (batched)
    # ------------------------------------------------------------------
    @staticmethod
    def _ry(angles: torch.Tensor) -> torch.Tensor:
        """Batch Ry matrix. angles: [B] → [B, 2, 2] complex64"""
        c = torch.cos(angles / 2).to(torch.complex64)
        s = torch.sin(angles / 2).to(torch.complex64)
        z = torch.zeros_like(c)
        row0 = torch.stack([c, -s], dim=-1)
        row1 = torch.stack([s,  c], dim=-1)
        return torch.stack([row0, row1], dim=-2)   # [B, 2, 2]

    @staticmethod
    def _rz(angles: torch.Tensor) -> torch.Tensor:
        """Batch Rz matrix. angles: [B] → [B, 2, 2] complex64"""
        half = angles / 2
        e_neg = torch.exp(-1j * half).to(torch.complex64)
        e_pos = torch.exp(1j * half).to(torch.complex64)
        z = torch.zeros_like(e_neg)
        row0 = torch.stack([e_neg, z],    dim=-1)
        row1 = torch.stack([z,    e_pos], dim=-1)
        return torch.stack([row0, row1], dim=-2)   # [B, 2, 2]

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------
    def _apply_single(
        self, state: torch.Tensor, gate: torch.Tensor, qubit: int
    ) -> torch.Tensor:
        """
        Apply a batched single-qubit gate.
        state: [B, 2^n] complex64
        gate:  [B, 2, 2] complex64  OR  [2, 2] (broadcast)
        """
        n = self.n_qubits
        B = state.shape[0]
        # Reshape: [B, 2, 2, ..., 2]
        s = state.reshape([B] + [2] * n)

        # Permute target qubit to dimension 1 (after batch)
        dim = qubit + 1
        perm = list(range(n + 1))
        perm[1], perm[dim] = perm[dim], perm[1]
        s = s.permute(perm).contiguous()           # [B, 2, ...]

        # Flatten non-target dims: [B, 2, rest]
        rest = 2 ** (n - 1)
        s = s.reshape(B, 2, rest)

        if gate.dim() == 3:
            # Batched gate [B, 2, 2]
            s = torch.einsum("bij,bjk->bik", gate, s)
        else:
            # Shared gate [2, 2]
            s = torch.einsum("ij,bjk->bik", gate, s)

        # Reshape back and unpermute
        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        s = s.reshape([B] + [2] * n)
        s = s.permute(inv_perm).contiguous()
        return s.reshape(B, self.n_states)

    def _apply_cnot(self, state: torch.Tensor, ctrl: int, tgt: int) -> torch.Tensor:
        """Apply CNOT using precomputed index masks. state: [B, 2^n]"""
        src = getattr(self, f"cnot_{ctrl}_{tgt}_src")
        dst = getattr(self, f"cnot_{ctrl}_{tgt}_dst")
        out = state.clone()
        out[:, src] = state[:, dst]
        out[:, dst] = state[:, src]
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_qubits] float32 — angle-embedded features
        returns: [B, n_qubits] float32 — PauliZ expectation values
        """
        B = x.shape[0]
        device = x.device
        dtype = torch.complex64

        # |0...0⟩ initial state
        state = torch.zeros(B, self.n_states, dtype=dtype, device=device)
        state[:, 0] = 1.0 + 0j

        # AngleEmbedding: Ry(x_i) on qubit i
        for i in range(self.n_qubits):
            gate = self._ry(x[:, i])      # [B, 2, 2]
            state = self._apply_single(state, gate, i)

        # StronglyEntanglingLayers
        for layer in range(self.n_layers):
            w = self.theta[layer]          # [n_qubits, 3]

            # Rotation: Rz(t0) Ry(t1) Rz(t2) per qubit — shared (not batched per sample)
            for q in range(self.n_qubits):
                t0, t1, t2 = w[q, 0], w[q, 1], w[q, 2]
                gz1 = self._rz(t0.unsqueeze(0)).squeeze(0)   # [2, 2]
                gy  = self._ry(t1.unsqueeze(0)).squeeze(0)
                gz2 = self._rz(t2.unsqueeze(0)).squeeze(0)
                state = self._apply_single(state, gz1, q)
                state = self._apply_single(state, gy,  q)
                state = self._apply_single(state, gz2, q)

            # CNOT ring entanglement
            for q in range(self.n_qubits):
                state = self._apply_cnot(state, q, (q + 1) % self.n_qubits)

        # PauliZ measurement: ⟨Z_i⟩ = P(|0⟩_i) - P(|1⟩_i)
        probs = state.abs().pow(2)        # [B, 2^n]
        s = probs.reshape(B, *([2] * self.n_qubits))
        expectations = []
        for i in range(self.n_qubits):
            p0 = s.select(i + 1, 0).reshape(B, -1).sum(-1)
            p1 = s.select(i + 1, 1).reshape(B, -1).sum(-1)
            expectations.append(p0 - p1)
        return torch.stack(expectations, dim=-1)   # [B, n_qubits]

    def get_unitary_matrices(self) -> dict:
        """
        Extract learned unitary matrices for FPGA export.
        Returns dict: layer → qubit → 2x2 numpy complex array
        """
        import numpy as np
        result = {}
        with torch.no_grad():
            for layer in range(self.n_layers):
                result[f"layer_{layer}"] = {}
                w = self.theta[layer]
                for q in range(self.n_qubits):
                    t0, t1, t2 = w[q, 0].item(), w[q, 1].item(), w[q, 2].item()

                    def rz(t):
                        return np.array([
                            [math.cos(t/2) - 1j*math.sin(t/2), 0],
                            [0, math.cos(t/2) + 1j*math.sin(t/2)]
                        ], dtype=complex)

                    def ry(t):
                        return np.array([
                            [math.cos(t/2), -math.sin(t/2)],
                            [math.sin(t/2),  math.cos(t/2)]
                        ], dtype=complex)

                    # Composed gate: Rz(t2) @ Ry(t1) @ Rz(t0)
                    U = rz(t2) @ ry(t1) @ rz(t0)
                    result[f"layer_{layer}"][f"qubit_{q}"] = U.tolist()
        return result
