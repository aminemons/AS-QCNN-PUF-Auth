"""
Pure TensorFlow GPU-native Variational Quantum Circuit.

KEY FIX: CNOT index tensors are stored as plain numpy arrays during build()
(Keras runs build() in a temporary scratch_graph to compute output shapes).
They are converted to tf.constant() INSIDE call(), which is traced in the
real graph and therefore always in scope.
"""

import tensorflow as tf
import numpy as np
import math


class QuantumKerasLayer(tf.keras.layers.Layer):
    """
    Variational Quantum Circuit implemented entirely in TensorFlow.
    Supports CPU and GPU with full batch parallelism.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.n_layers = n_layers
        # Pre-compute CNOT swap indices as NUMPY (not tf.constant) so they
        # survive the Keras scratch_graph that runs during build().
        self._precompute_cnot_np()

    def _precompute_cnot_np(self):
        """Store swap pairs as numpy arrays — graph-safe."""
        n = self.n_qubits
        N = self.n_states
        self._src_np = {}
        self._dst_np = {}
        for ctrl in range(n):
            tgt = (ctrl + 1) % n
            idx = np.arange(N, dtype=np.int32)
            ctrl_high = ((idx >> (n - 1 - ctrl)) & 1) == 1
            flip = np.int32(1 << (n - 1 - tgt))
            self._src_np[(ctrl, tgt)] = idx[ctrl_high]          # shape [K]
            self._dst_np[(ctrl, tgt)] = (idx ^ flip)[ctrl_high] # shape [K]

    def build(self, input_shape):
        init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        self.theta = self.add_weight(
            name="vqc_weights",
            shape=(self.n_layers, self.n_qubits, 3),
            initializer=init,
            trainable=True,
        )
        super().build(input_shape)

    # ── Gate factories ──────────────────────────────────────────────────────────

    def _h(self):
        inv_sqrt2 = tf.cast(1.0 / math.sqrt(2), tf.complex64)
        return tf.stack([
            [inv_sqrt2,  inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2],
        ])

    def _ry(self, angles):
        """angles: float32 [B] or scalar → returns [..., 2, 2] complex64"""
        angles = tf.cast(angles, tf.float32)
        c = tf.cast(tf.cos(angles / 2.0), tf.complex64)
        s = tf.cast(tf.sin(angles / 2.0), tf.complex64)
        return tf.stack([tf.stack([c, -s], axis=-1),
                         tf.stack([s,  c], axis=-1)], axis=-2)

    def _rz(self, angles):
        """angles: float32 [B] or scalar → returns [..., 2, 2] complex64"""
        angles = tf.cast(angles, tf.float32)
        half   = angles / 2.0
        c = tf.math.cos(half)
        s = tf.math.sin(half)
        e_neg = tf.complex(c, -s)
        e_pos = tf.complex(c,  s)
        z = tf.zeros_like(e_neg)
        return tf.stack([tf.stack([e_neg, z], axis=-1),
                         tf.stack([z, e_pos], axis=-1)], axis=-2)

    # ── State helpers ───────────────────────────────────────────────────────────

    def _apply_single(self, state, gate, qubit):
        """
        state: [B, N]   gate: [2,2] or [B,2,2]
        """
        n  = self.n_qubits
        B  = tf.shape(state)[0]
        # [B, 2, 2, ..., 2]
        s = tf.reshape(state, tf.concat([[B], [2]*n], axis=0))
        # Move target qubit axis to position 1
        dim = qubit + 1
        perm = list(range(n + 1))
        perm[1], perm[dim] = perm[dim], perm[1]
        s = tf.transpose(s, perm=perm)
        # [B, 2, rest]
        rest = 2 ** (n - 1)
        s = tf.reshape(s, [B, 2, rest])
        if len(gate.shape) == 3:
            s = tf.einsum("bij,bjk->bik", gate, s)
        else:
            s = tf.einsum("ij,bjk->bik", gate, s)
        # Inverse permutation
        inv = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv[p] = i
        s = tf.reshape(s, tf.concat([[B], [2]*n], axis=0))
        s = tf.transpose(s, perm=inv)
        return tf.reshape(s, [B, self.n_states])

    def _apply_cnot(self, state, ctrl, tgt):
        """
        CNOT gate via gather/scatter.
        Converts numpy index arrays to tf.constant HERE (inside call()),
        so they are always created in the active graph — not a scratch graph.
        """
        src = tf.constant(self._src_np[(ctrl, tgt)], dtype=tf.int32)  # [K]
        dst = tf.constant(self._dst_np[(ctrl, tgt)], dtype=tf.int32)  # [K]

        amp_src = tf.gather(state, src, axis=1)   # [B, K]
        amp_dst = tf.gather(state, dst, axis=1)   # [B, K]

        # Transpose trick: scatter along axis-0 (state index) of [N, B]
        st = tf.transpose(state)   # [N, B]
        st = tf.tensor_scatter_nd_update(st, tf.expand_dims(src, 1), tf.transpose(amp_dst))
        st = tf.tensor_scatter_nd_update(st, tf.expand_dims(dst, 1), tf.transpose(amp_src))
        return tf.transpose(st)    # [B, N]

    # ── Forward ─────────────────────────────────────────────────────────────────

    def call(self, x):
        """
        x: [B, n_qubits] float32 — pre-encoded rotation angles.
        Returns: [B, n_qubits] float32 — Pauli-Z expectation values.
        """
        B = tf.shape(x)[0]

        # |0…0⟩
        state = tf.zeros([B, self.n_states], dtype=tf.complex64)
        idx0  = tf.stack([tf.range(B), tf.zeros(B, dtype=tf.int32)], axis=1)
        state = tf.tensor_scatter_nd_update(state, idx0, tf.ones([B], tf.complex64))

        # 1. ZZ Feature Map
        H = self._h()
        for i in range(self.n_qubits):
            state = self._apply_single(state, H, i)

        for i in range(self.n_qubits):
            state = self._apply_single(state, self._rz(x[:, i]), i)

        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            state = self._apply_cnot(state, i, j)
            state = self._apply_single(state, self._rz(x[:, i] * x[:, j]), j)
            state = self._apply_cnot(state, i, j)

        # 2. Strongly-entangling variational layers
        for layer in range(self.n_layers):
            w = self.theta[layer]   # [n_qubits, 3]
            for q in range(self.n_qubits):
                t0, t1, t2 = w[q, 0], w[q, 1], w[q, 2]
                # Scalar weights → expand to [1] so gate returns [1,2,2], then squeeze
                gz1 = self._rz(tf.expand_dims(t0, 0))[0]
                gy  = self._ry(tf.expand_dims(t1, 0))[0]
                gz2 = self._rz(tf.expand_dims(t2, 0))[0]
                state = self._apply_single(state, gz1, q)
                state = self._apply_single(state, gy,  q)
                state = self._apply_single(state, gz2, q)
            for q in range(self.n_qubits):
                state = self._apply_cnot(state, q, (q + 1) % self.n_qubits)

        # 3. Pauli-Z expectations  (real/imag avoids complex→float cast warnings)
        re = tf.math.real(state)
        im = tf.math.imag(state)
        probs = re * re + im * im                   # [B, N]  float32
        s = tf.reshape(probs, tf.concat([[B], [2]*self.n_qubits], axis=0))
        exps = []
        for i in range(self.n_qubits):
            axes = [a for a in range(1, self.n_qubits + 1) if a != i + 1]
            marg = tf.reduce_sum(s, axis=axes)   # [B, 2]
            exps.append(marg[:, 0] - marg[:, 1])
        return tf.stack(exps, axis=-1)           # [B, n_qubits]


def create_quantum_layer(n_qubits: int = 8, n_layers: int = 6) -> QuantumKerasLayer:
    return QuantumKerasLayer(n_qubits=n_qubits, n_layers=n_layers)


def get_unitary_matrices(weights: np.ndarray) -> dict:
    """
    Extract learned unitary matrices for FPGA export.
    weights: (n_layers, n_qubits, 3)
    """
    n_layers, n_qubits, _ = weights.shape
    result = {}
    for layer in range(n_layers):
        result[f"layer_{layer}"] = {}
        for q in range(n_qubits):
            t0, t1, t2 = weights[layer, q]

            def rz(t):
                c, s = math.cos(t/2), math.sin(t/2)
                return np.array([[c - 1j*s, 0], [0, c + 1j*s]], dtype=complex)

            def ry(t):
                c, s = math.cos(t/2), math.sin(t/2)
                return np.array([[c, -s], [s, c]], dtype=complex)

            result[f"layer_{layer}"][f"qubit_{q}"] = (rz(t2) @ ry(t1) @ rz(t0)).tolist()
    return result
