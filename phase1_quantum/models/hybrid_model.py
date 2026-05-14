"""
Hybrid Classical-Quantum model for PUF binary authentication in TensorFlow.

Pipeline:
  [64-bit challenge] → ClassicalEncoder(64→128→64→8)
                     → KerasLayer(8 qubits, 6 layers)
                     → ClassicalHead(8→1)
                     → sigmoid → binary response
"""

import tensorflow as tf
from .quantum_circuit import create_quantum_layer
import math

class ClassicalEncoder(tf.keras.layers.Layer):
    """Maps a PUF challenge patch to qubit rotation angles."""
    def __init__(self, in_features: int = 16, hidden: int = 128, out_features: int = 8):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        self.dense2 = tf.keras.layers.Dense(64)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        
        self.dense3 = tf.keras.layers.Dense(out_features)
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.dense3(x)
        x = self.tanh(x)
        return x * math.pi  # scale to [-pi, pi]


class ClassicalHead(tf.keras.layers.Layer):
    """Maps concatenated PauliZ expectation values to a binary prediction."""
    def __init__(self, in_features: int = 8):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(16)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)


class HybridQCNN(tf.keras.Model):
    """
    End-to-End Hybrid Quantum Convolutional Neural Network.
    Processes the PUF challenge in overlapping patches.
    """
    def __init__(self, n_bits: int = 64, n_qubits: int = 8, n_layers: int = 6):
        super().__init__()
        self.n_bits = n_bits
        self.n_qubits = n_qubits
        
        # We split the 64-bit challenge into 4 patches of 16 bits.
        self.patch_size = 16
        self.n_patches = n_bits // self.patch_size
        
        # Weight sharing across patches (like a CNN)
        self.encoder = ClassicalEncoder(in_features=self.patch_size, hidden=128, out_features=n_qubits)
        self.vqc = create_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)
        
        # Combine all patches expectation values
        self.head = ClassicalHead(in_features=self.n_patches * n_qubits)

    def call(self, x, training=False):
        # x is [B, 64]
        # Reshape to [B, 4, 16]
        patches = tf.reshape(x, [-1, self.n_patches, self.patch_size])
        
        # Process each patch
        patch_expectations = []
        for i in range(self.n_patches):
            patch = patches[:, i, :]
            angles = self.encoder(patch, training=training)
            # vqc layer
            expectations = self.vqc(angles)
            patch_expectations.append(expectations)
            
        # Concatenate across the feature dimension -> [B, 32]
        concat_exp = tf.concat(patch_expectations, axis=-1)
        
        logits = self.head(concat_exp, training=training)
        return logits

    def count_parameters_dict(self):
        """Helper to get counts for logging"""
        # Call model on dummy input to build weights
        self(tf.zeros((1, self.n_bits)))
        total = sum([tf.keras.backend.count_params(w) for w in self.trainable_weights])
        return {"total": total}

