"""
PennyLane-based Variational Quantum Circuit for TensorFlow.
Integrates seamlessly with TensorFlow Keras models and PyGAD.
"""

import tensorflow as tf
import pennylane as qml
import numpy as np
import math


def create_quantum_layer(n_qubits: int = 8, n_layers: int = 6):
    """
    Creates a TensorFlow KerasLayer containing the Quantum Circuit.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="tf")
    def qnode(inputs, weights):
        # inputs:  [n_qubits]
        # weights: [n_layers, n_qubits, 3]

        # 1. ZZ Feature Map
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_qubits):
            qml.RZ(inputs[i], wires=i)

        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])
            qml.RZ(inputs[i] * inputs[j], wires=j)
            qml.CNOT(wires=[i, j])

        # 2. Strongly Entangling Layers
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RZ(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    
    # Random uniform initialization logic similar to PyTorch
    init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
    
    qlayer = qml.qnn.KerasLayer(
        qnode, 
        weight_shapes, 
        output_dim=n_qubits,
        weight_specs={"weights": {"initializer": init}}
    )
    return qlayer


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
