"""
Pure TensorFlow GPU-native Variational Quantum Circuit.
Replaces PennyLane KerasLayer (which is deprecated/buggy in Keras 3) with blazing fast batched GPU matrix multiplications.
"""

import tensorflow as tf
import numpy as np
import math

class QuantumKerasLayer(tf.keras.layers.Layer):
    """
    Variational Quantum Circuit implemented entirely in TensorFlow.
    Runs on CUDA with full batch parallelism.
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
        self._register_cnot_masks()
        super().build(input_shape)

    def _register_cnot_masks(self):
        n = self.n_qubits
        self.cnot_masks = {}
        for ctrl in range(n):
            tgt = (ctrl + 1) % n
            indices = np.arange(2 ** n)
            ctrl_is_1 = (indices >> (n - 1 - ctrl)) & 1 == 1
            flip_mask = 1 << (n - 1 - tgt)
            flipped = indices ^ flip_mask
            self.cnot_masks[f"{ctrl}_{tgt}_src"] = indices[ctrl_is_1].astype(np.int32)
            self.cnot_masks[f"{ctrl}_{tgt}_dst"] = flipped[ctrl_is_1].astype(np.int32)

    def _h(self):
        inv_sqrt2 = 1.0 / math.sqrt(2)
        return tf.constant([
            [inv_sqrt2,  inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2]
        ], dtype=tf.complex64)

    def _ry(self, angles):
        angles = tf.cast(angles, tf.float32)
        c = tf.cast(tf.math.cos(angles / 2.0), tf.complex64)
        s = tf.cast(tf.math.sin(angles / 2.0), tf.complex64)
        row0 = tf.stack([c, -s], axis=-1)
        row1 = tf.stack([s,  c], axis=-1)
        return tf.stack([row0, row1], axis=-2)

    def _rz(self, angles):
        angles = tf.cast(angles, tf.float32)
        half = angles / 2.0
        c = tf.math.cos(half)
        s = tf.math.sin(half)
        e_neg = tf.complex(c, -s)
        e_pos = tf.complex(c, s)
        z = tf.zeros_like(e_neg)
        row0 = tf.stack([e_neg, z], axis=-1)
        row1 = tf.stack([z, e_pos], axis=-1)
        return tf.stack([row0, row1], axis=-2)

    def _apply_single(self, state, gate, qubit):
        n = self.n_qubits
        B = tf.shape(state)[0]
        
        # Reshape: [B, 2, 2, ..., 2]
        s = tf.reshape(state, [B] + [2] * n)
        
        # Permute target qubit to dimension 1 (after batch)
        dim = qubit + 1
        perm = list(range(n + 1))
        perm[1], perm[dim] = perm[dim], perm[1]
        s = tf.transpose(s, perm=perm)
        
        # Flatten non-target dims: [B, 2, rest]
        rest = 2 ** (n - 1)
        s = tf.reshape(s, [B, 2, rest])
        
        if len(gate.shape) == 3:
            s = tf.einsum("bij,bjk->bik", gate, s)
        else:
            s = tf.einsum("ij,bjk->bik", gate, s)
            
        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        s = tf.reshape(s, [B] + [2] * n)
        s = tf.transpose(s, perm=inv_perm)
        return tf.reshape(s, [B, self.n_states])

    def _apply_cnot(self, state, ctrl, tgt):
        src = tf.constant(self.cnot_masks[f"{ctrl}_{tgt}_src"], dtype=tf.int32)
        dst = tf.constant(self.cnot_masks[f"{ctrl}_{tgt}_dst"], dtype=tf.int32)
        
        state_src = tf.gather(state, src, axis=1)
        state_dst = tf.gather(state, dst, axis=1)
        
        # In TF, we cannot do inplace updates like state[:, src] = state_dst easily.
        # We must use tensor_scatter_nd_update
        B = tf.shape(state)[0]
        
        # Create indices for scatter
        b_idx = tf.repeat(tf.range(B), tf.shape(src)[0])
        src_idx = tf.tile(src, [B])
        dst_idx = tf.tile(dst, [B])
        
        # Shape: [B * len(src), 2]
        indices_src = tf.stack([b_idx, src_idx], axis=1)
        indices_dst = tf.stack([b_idx, dst_idx], axis=1)
        
        out = tf.tensor_scatter_nd_update(state, indices_src, tf.reshape(state_dst, [-1]))
        out = tf.tensor_scatter_nd_update(out, indices_dst, tf.reshape(state_src, [-1]))
        return out

    def call(self, x):
        B = tf.shape(x)[0]
        
        # |0...0⟩ initial state
        state = tf.zeros([B, self.n_states], dtype=tf.complex64)
        indices = tf.stack([tf.range(B), tf.zeros(B, dtype=tf.int32)], axis=1)
        updates = tf.ones([B], dtype=tf.complex64)
        state = tf.tensor_scatter_nd_update(state, indices, updates)

        # 1. ZZ Feature Map
        h_gate = self._h()
        for i in range(self.n_qubits):
            state = self._apply_single(state, h_gate, i)

        for i in range(self.n_qubits):
            gate = self._rz(x[:, i])
            state = self._apply_single(state, gate, i)

        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            theta = x[:, i] * x[:, j]
            state = self._apply_cnot(state, i, j)
            gate = self._rz(theta)
            state = self._apply_single(state, gate, j)
            state = self._apply_cnot(state, i, j)

        # 2. StronglyEntanglingLayers
        for layer in range(self.n_layers):
            w = self.theta[layer]
            for q in range(self.n_qubits):
                t0, t1, t2 = w[q, 0], w[q, 1], w[q, 2]
                gz1 = tf.squeeze(self._rz(tf.expand_dims(t0, 0)), axis=0)
                gy  = tf.squeeze(self._ry(tf.expand_dims(t1, 0)), axis=0)
                gz2 = tf.squeeze(self._rz(tf.expand_dims(t2, 0)), axis=0)
                state = self._apply_single(state, gz1, q)
                state = self._apply_single(state, gy,  q)
                state = self._apply_single(state, gz2, q)

            for q in range(self.n_qubits):
                state = self._apply_cnot(state, q, (q + 1) % self.n_qubits)

        # Measurement
        probs = tf.math.square(tf.math.abs(state))
        s = tf.reshape(probs, [B] + [2] * self.n_qubits)
        
        expectations = []
        for i in range(self.n_qubits):
            # To marginalize over qubit `i`, we sum over all other axes.
            # In TF, we want to sum over axes [1..n_qubits] EXCEPT (i+1).
            axes_to_sum = [a for a in range(1, self.n_qubits + 1) if a != (i + 1)]
            
            # Sum out all non-target qubits
            marginal = tf.reduce_sum(s, axis=axes_to_sum)  # Shape [B, 2]
            
            # Expectation value is P(|0>) - P(|1>)
            p0 = marginal[:, 0]
            p1 = marginal[:, 1]
            expectations.append(p0 - p1)
            
        return tf.stack(expectations, axis=-1)

def create_quantum_layer(n_qubits: int = 8, n_layers: int = 6):
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
