"""
Hybrid Classical-Quantum model for PUF binary authentication in TensorFlow.

Architecture (from GA search, best=77.4% in 7 epochs):
  [64-bit challenge] → 16 patches of 4 bits
                     → ClassicalEncoder(4→32→4) per patch
                     → QuantumKerasLayer(4 qubits, 5 layers)
                     → ClassicalHead(64→64→64→1)
                     → sigmoid → binary response
"""

import tensorflow as tf
from .quantum_circuit import create_quantum_layer
import math


class ClassicalEncoder(tf.keras.layers.Layer):
    """Maps a PUF challenge patch to qubit rotation angles."""
    def __init__(self, in_features: int = 4, hidden_layers=None, out_features: int = 4, dropout: float = 0.193):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32]

        layers = []
        prev = in_features
        for h in hidden_layers:
            layers.append(tf.keras.layers.Dense(h))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.ReLU())
            layers.append(tf.keras.layers.Dropout(dropout))
            prev = h

        layers.append(tf.keras.layers.Dense(out_features))
        layers.append(tf.keras.layers.Activation('tanh'))
        self._enc_layers = layers

    def call(self, x, training=False):
        for layer in self._enc_layers:
            if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x * math.pi  # scale to [-π, π]


class ClassicalHead(tf.keras.layers.Layer):
    """Maps concatenated PauliZ expectation values to a binary prediction."""
    def __init__(self, in_features: int = 64, hidden_layers=None, dropout: float = 0.193):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]

        layers = []
        for h in hidden_layers:
            layers.append(tf.keras.layers.Dense(h))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.ReLU())
            layers.append(tf.keras.layers.Dropout(dropout))

        layers.append(tf.keras.layers.Dense(1))
        self._head_layers = layers

    def call(self, x, training=False):
        for layer in self._head_layers:
            if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class HybridQCNN(tf.keras.Model):
    """
    End-to-End Hybrid Quantum Convolutional Neural Network.
    Processes the PUF challenge in overlapping patches.

    Default architecture matches GA-optimized result:
      n_qubits=4, vqc_layers=5, n_patches=16, enc=[32], head=[64,64], drop=0.193
    """
    def __init__(self, n_bits: int = 64, n_qubits: int = 4, n_layers: int = 5,
                 n_patches: int = 16, encoder_hidden=None, head_hidden=None,
                 dropout: float = 0.193):
        super().__init__()
        self.n_bits = n_bits
        self.n_qubits = n_qubits
        self.n_patches = n_patches
        self.patch_size = n_bits // n_patches   # 64/16 = 4

        if encoder_hidden is None:
            encoder_hidden = [32]
        if head_hidden is None:
            head_hidden = [64, 64]

        # Weight sharing across patches (like a CNN)
        self.encoder = ClassicalEncoder(
            in_features=self.patch_size,
            hidden_layers=encoder_hidden,
            out_features=n_qubits,
            dropout=dropout,
        )
        self.vqc = create_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)

        # Combine all patches' expectation values
        self.head = ClassicalHead(
            in_features=self.n_patches * n_qubits,
            hidden_layers=head_hidden,
            dropout=dropout,
        )

    def call(self, x, training=False):
        # x: [B, 64]
        patches = tf.reshape(x, [-1, self.n_patches, self.patch_size])  # [B, 16, 4]

        patch_expectations = []
        for i in range(self.n_patches):
            patch = patches[:, i, :]          # [B, 4]
            angles = self.encoder(patch, training=training)  # [B, n_qubits]
            expectations = self.vqc(angles)   # [B, n_qubits]
            patch_expectations.append(expectations)

        # [B, 16*4] = [B, 64]
        concat_exp = tf.concat(patch_expectations, axis=-1)
        return self.head(concat_exp, training=training)   # [B, 1]

    def count_parameters_dict(self):
        """Helper to get counts for logging."""
        self(tf.zeros((1, self.n_bits)))
        total = sum(tf.keras.backend.count_params(w) for w in self.trainable_weights)
        return {"total": total}
