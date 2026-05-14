"""
Pure TensorFlow GPU-native Variational Quantum Circuit.
Replaces PennyLane KerasLayer (deprecated in Keras 3) with GPU matrix multiplications.

All operations are @tf.function compatible — no Python-level loops over batch dim.
"""

import tensorflow as tf
import numpy as np
import math


class QuantumKerasLayer(tf.keras.layers.Layer):
    """
    Variational Quantum Circuit implemented entirely in TensorFlow.
    Runs on CUDA with full batch parallelism.
    All graph-mode (tf.function) compatible.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.n_layers = n_layers

    def build(self, input_shape):
        init = tf.keras.initializers.RandomUniform(minval=-1e-4, maxval=1e-4)
        self.theta = self.add_weight(
            name="quantum_weights",
            shape=(self.n_layers, self.n_qubits, 3),
            initializer=init,
            trainable=True
        )
        # Pre-compute static CNOT swap pairs as tf.constant (not Python dicts)
        self._build_cnot_tensors()
        super().build(input_shape)

    def _build_cnot_tensors(self):
        """
        For each CNOT(ctrl, tgt), pre-compute which state-indices need to be swapped.
        Stored as tf.constant tensors so they live in the graph permanently.
        """
        n = self.n_qubits
        N = self.n_states
        self._cnot_src = {}
        self._cnot_dst = {}
        for ctrl in range(n):
            tgt = (ctrl + 1) % n
            indices = np.arange(N, dtype=np.int32)
            ctrl_is_1 = ((indices >> (n - 1 - ctrl)) & 1) == 1
            flip_mask = np.int32(1 << (n - 1 - tgt))
            flipped = indices ^ flip_mask
            src = tf.constant(indices[ctrl_is_1], dtype=tf.int32)
            dst = tf.constant(flipped[ctrl_is_1], dtype=tf.int32)
            self._cnot_src[(ctrl, tgt)] = src
            self._cnot_dst[(ctrl, tgt)] = dst

    # ── Gate factory methods ────────────────────────────────────────────────────

    def _h(self):
        inv_sqrt2 = 1.0 / math.sqrt(2)
        return tf.constant([
            [inv_sqrt2,  inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2]
        ], dtype=tf.complex64)

    def _ry(self, angles):
        """angles: float32 scalar or 1-D tensor [B]"""
        angles = tf.cast(angles, tf.float32)
        c = tf.cast(tf.math.cos(angles / 2.0), tf.complex64)
        s = tf.cast(tf.math.sin(angles / 2.0), tf.complex64)
        row0 = tf.stack([c, -s], axis=-1)
        row1 = tf.stack([s,  c], axis=-1)
        return tf.stack([row0, row1], axis=-2)   # [..., 2, 2]

    def _rz(self, angles):
        """angles: float32 scalar or 1-D tensor [B]"""
        angles = tf.cast(angles, tf.float32)
        half = angles / 2.0
        c = tf.math.cos(half)
        s = tf.math.sin(half)
        e_neg = tf.complex(c, -s)
        e_pos = tf.complex(c,  s)
        z = tf.zeros_like(e_neg)
        row0 = tf.stack([e_neg, z], axis=-1)
        row1 = tf.stack([z, e_pos], axis=-1)
        return tf.stack([row0, row1], axis=-2)   # [..., 2, 2]

    # ── State evolution helpers ─────────────────────────────────────────────────

    def _apply_single(self, state, gate, qubit):
        """
        Apply a 2×2 gate to `qubit` of the full statevector.
        state: [B, N]   gate: [2,2] (shared) or [B,2,2] (per-sample)
        """
        n = self.n_qubits
        B = tf.shape(state)[0]

        # Reshape to [B, 2, 2, ..., 2]  (n qubit dims after batch)
        s = tf.reshape(state, tf.concat([[B], [2] * n], axis=0))

        # Permute target qubit to axis 1 (right after batch)
        dim = qubit + 1
        perm = list(range(n + 1))
        perm[1], perm[dim] = perm[dim], perm[1]
        s = tf.transpose(s, perm=perm)

        # Flatten non-target dims: [B, 2, rest]
        rest = 2 ** (n - 1)
        s = tf.reshape(s, [B, 2, rest])

        if len(gate.shape) == 3:          # per-sample gate [B,2,2]
            s = tf.einsum("bij,bjk->bik", gate, s)
        else:                             # shared gate [2,2]
            s = tf.einsum("ij,bjk->bik", gate, s)

        # Inverse permutation
        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i

        s = tf.reshape(s, tf.concat([[B], [2] * n], axis=0))
        s = tf.transpose(s, perm=inv_perm)
        return tf.reshape(s, [B, self.n_states])

    def _apply_cnot(self, state, ctrl, tgt):
        """
        Swap amplitude entries for CNOT(ctrl→tgt).
        Uses pure gather/scatter — fully graph-mode safe.
        state: [B, N]
        """
        src = self._cnot_src[(ctrl, tgt)]   # [K]  static int32
        dst = self._cnot_dst[(ctrl, tgt)]   # [K]  static int32

        amp_at_src = tf.gather(state, src, axis=1)   # [B, K]
        amp_at_dst = tf.gather(state, dst, axis=1)   # [B, K]

        # Scatter amp_at_dst into src positions, amp_at_src into dst positions
        # tf.tensor_scatter_nd_update expects indices shape [num_updates, rank]
        # For a 2-D state [B, N], updates along axis=1 → use transpose trick:
        state_T = tf.transpose(state)          # [N, B]
        state_T = tf.tensor_scatter_nd_update(
            state_T,
            tf.expand_dims(src, 1),            # [K, 1]
            tf.transpose(amp_at_dst)           # [K, B]
        )
        state_T = tf.tensor_scatter_nd_update(
            state_T,
            tf.expand_dims(dst, 1),            # [K, 1]
            tf.transpose(amp_at_src)           # [K, B]
        )
        return tf.transpose(state_T)           # [B, N]

    # ── Forward pass ────────────────────────────────────────────────────────────

    def call(self, x):
        """
        x: [B, n_qubits]  float32 — pre-encoded rotation angles from ClassicalEncoder.
        Returns: [B, n_qubits]  float32 — Pauli-Z expectation values.
        """
        B = tf.shape(x)[0]

        # |0…0⟩ initial state
        state = tf.zeros([B, self.n_states], dtype=tf.complex64)
        indices = tf.stack([tf.range(B), tf.zeros(B, dtype=tf.int32)], axis=1)
        updates = tf.ones([B], dtype=tf.complex64)
        state = tf.tensor_scatter_nd_update(state, indices, updates)

        # ── 1. ZZ Feature Map ──────────────────────────────────────────────────
        h_gate = self._h()
        for i in range(self.n_qubits):
            state = self._apply_single(state, h_gate, i)

        for i in range(self.n_qubits):
            gate = self._rz(x[:, i])            # [B, 2, 2]
            state = self._apply_single(state, gate, i)

        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            theta = x[:, i] * x[:, j]           # [B]
            state = self._apply_cnot(state, i, j)
            gate = self._rz(theta)               # [B, 2, 2]
            state = self._apply_single(state, gate, j)
            state = self._apply_cnot(state, i, j)

        # ── 2. StronglyEntanglingLayers ────────────────────────────────────────
        for layer in range(self.n_layers):
            w = self.theta[layer]                # [n_qubits, 3]
            for q in range(self.n_qubits):
                t0, t1, t2 = w[q, 0], w[q, 1], w[q, 2]
                # Expand scalar weights to [1] so _rz/_ry return [1,2,2], then squeeze
                gz1 = tf.squeeze(self._rz(tf.expand_dims(t0, 0)), axis=0)   # [2,2]
                gy  = tf.squeeze(self._ry(tf.expand_dims(t1, 0)), axis=0)
                gz2 = tf.squeeze(self._rz(tf.expand_dims(t2, 0)), axis=0)
                state = self._apply_single(state, gz1, q)
                state = self._apply_single(state, gy,  q)
                state = self._apply_single(state, gz2, q)

            for q in range(self.n_qubits):
                state = self._apply_cnot(state, q, (q + 1) % self.n_qubits)

        # ── 3. Measurement (Pauli-Z expectation) ──────────────────────────────
        probs = tf.math.square(tf.math.abs(state))   # [B, N]
        s = tf.reshape(probs, tf.concat([[B], [2] * self.n_qubits], axis=0))

        expectations = []
        for i in range(self.n_qubits):
            axes_to_sum = [a for a in range(1, self.n_qubits + 1) if a != (i + 1)]
            marginal = tf.reduce_sum(s, axis=axes_to_sum)   # [B, 2]
            p0 = marginal[:, 0]
            p1 = marginal[:, 1]
            expectations.append(p0 - p1)

        return tf.stack(expectations, axis=-1)   # [B, n_qubits]


def create_quantum_layer(n_qubits: int = 8, n_layers: int = 6) -> "QuantumKerasLayer":
    return QuantumKerasLayer(n_qubits=n_qubits, n_layers=n_layers)


def get_unitary_matrices(weights: np.ndarray) -> dict:
    """
    Extract learned unitary matrices for FPGA export.
    weights shape: (n_layers, n_qubits, 3)
    Returns dict: layer -> qubit -> 2x2 numpy complex array
    """
    n_layers, n_qubits, _ = weights.shape
    result = {}

    for layer in range(n_layers):
        result[f"layer_{layer}"] = {}
        for q in range(n_qubits):
            t0, t1, t2 = weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2]

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
