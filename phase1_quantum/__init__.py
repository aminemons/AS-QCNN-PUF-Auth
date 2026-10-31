"""QCNN package; load TensorFlow only when a model or trainer is requested."""
from importlib import import_module

__all__ = [
    "HybridQCNN",
    "GATrainer"
]


def __getattr__(name):
    modules = {"HybridQCNN": "phase1_quantum.models", "GATrainer": "phase1_quantum.training"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
