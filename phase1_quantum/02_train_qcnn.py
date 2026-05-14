"""
Main Training Script (TensorFlow + Genetic Algorithm)

Trains HybridQCNN on 1-XOR, 3-XOR, 5-XOR Arbiter PUF datasets.
Uses PyGAD evolutionary optimization to bypass local minima and maximize accuracy.
Exports unitary matrices directly to fixed-point FPGA headers.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_quantum.models import HybridQCNN
from phase1_quantum.training import GATrainer
from phase1_quantum.models.quantum_circuit import get_unitary_matrices
from phase1_quantum.export.fixed_point import quantize_matrix, matrix_to_hls_cpp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_tf_dataset(npz_path, batch_size):
    """Loads NPZ into tf.data.Dataset"""
    data = np.load(npz_path)
    challenges = data["challenges"].astype(np.float32)
    responses = data["responses"].astype(np.int32)
    
    # 80/10/10 split
    n = len(challenges)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_ds = tf.data.Dataset.from_tensor_slices((challenges[:n_train], responses[:n_train]))
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((challenges[n_train:n_train+n_val], responses[n_train:n_train+n_val]))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((challenges[n_train+n_val:], responses[n_train+n_val:]))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def hardware_report():
    logger.info("=" * 60)
    logger.info("Hardware Report")
    logger.info(f"  TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"  GPUs available     : {len(gpus)}")
    for gpu in gpus:
        logger.info(f"    - {gpu.name}")
    logger.info("=" * 60)


def find_dataset(data_dir: Path, xor_k: int) -> Path:
    matches = list(data_dir.glob(f"puf_{xor_k}xor_*.npz"))
    if not matches:
        logger.error(f"No dataset found for {xor_k}-XOR in {data_dir}.")
        sys.exit(1)
    return sorted(matches)[-1]


def export_fpga_matrices(model, out_dir, tag):
    """Extracts weights from TF model, computes unitary matrices, and saves C++ headers."""
    # Find the KerasLayer in the HybridQCNN
    vqc_layer = model.vqc
    
    # Extract weights [n_layers, n_qubits, 3]
    vqc_weights = vqc_layer.weights[0].numpy()
    
    matrices = get_unitary_matrices(vqc_weights)
    hls_dir = out_dir / "fpga_headers" / tag
    hls_dir.mkdir(parents=True, exist_ok=True)
    
    for layer, qubits in matrices.items():
        for q, matrix_list in qubits.items():
            U = np.array(matrix_list, dtype=complex)
            var_name = f"unitary_{layer}_{q}"
            cpp_code = matrix_to_hls_cpp(U, var_name)
            
            with open(hls_dir / f"{var_name}.h", "w") as f:
                f.write(f"// Auto-generated Fixed-Point Matrix for FPGA (ap_fixed<16,4>)\n")
                f.write(cpp_code)
                
    logger.info(f"Exported {len(matrices) * len(matrices['layer_0'])} FPGA Unitary Headers to {hls_dir}")


def train_puf_type(xor_k: int, cfg: dict) -> dict:
    tag = f"{xor_k}xor"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training on {xor_k}-XOR Arbiter PUF with Genetic Algorithm")
    logger.info(f"{'='*60}")

    data_dir = Path(cfg.get("data_dir", "data"))
    npz_path = find_dataset(data_dir, xor_k)
    logger.info(f"Dataset: {npz_path}")

    batch_size = cfg.get("batch_size", 512)
    train_ds, val_ds, test_ds = load_tf_dataset(npz_path, batch_size)

    # Initialize TF Model
    model = HybridQCNN(
        n_bits=cfg.get("n_bits", 64),
        n_qubits=cfg.get("n_qubits", 8),
        n_layers=cfg.get("n_layers", 6),
    )
    
    # Must call model once to build weights before passing to GA
    dummy_input = tf.zeros((1, cfg.get("n_bits", 64)))
    model(dummy_input)
    param_counts = model.count_parameters_dict()
    logger.info(f"Model parameters: {param_counts}")

    trainer = GATrainer(
        model=model,
        cfg=cfg,
        run_tag=tag,
        output_dir=cfg.get("output_dir", "results"),
    )

    t0 = time.time()
    best_ckpt, history = trainer.fit(train_ds, val_ds)
    elapsed = time.time() - t0

    logger.info(f"Evaluating best model on full test set...")
    correct = 0
    total = 0
    for x, y in test_ds:
        logits = model(x, training=False)
        preds = tf.cast(logits > 0, tf.int32)
        correct += tf.reduce_sum(tf.cast(preds == tf.expand_dims(y, 1), tf.int32)).numpy()
        total += x.shape[0]
        
    test_acc = correct / total

    result = {
        "puf_type":  tag,
        "test_acc":  test_acc,
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
        "train_time_min": elapsed / 60,
        "best_ckpt": str(best_ckpt),
        "param_counts": param_counts,
    }

    logger.info(f"{tag} — Test accuracy: {test_acc:.4f} | Time: {elapsed/60:.1f} min")

    out_dir = Path(cfg.get("output_dir", "results"))
    export_fpga_matrices(model, out_dir, tag)

    return result


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid QCNN with GA")
    parser.add_argument("--config",    type=str, required=True,         help="Path to YAML config")
    parser.add_argument("--puf_types", type=str, default="1xor,3xor,5xor",
                        help="Comma-separated list: 1xor,3xor,5xor")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    hardware_report()

    puf_types = [x.strip() for x in args.puf_types.split(",")]
    xor_map = {"1xor": 1, "3xor": 3, "5xor": 5}

    all_results = []
    for puf_str in puf_types:
        if puf_str not in xor_map:
            logger.warning(f"Unknown PUF type: {puf_str}, skipping.")
            continue
        result = train_puf_type(xor_map[puf_str], cfg)
        all_results.append(result)

    # Save summary
    out_dir = Path(cfg.get("output_dir", "results")) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nTraining complete. Summary saved: {summary_path}")
    logger.info("\nResults:")
    for r in all_results:
        logger.info(f"  {r['puf_type']:8s} → test_acc={r['test_acc']:.4f}  time={r['train_time_min']:.1f}min")


if __name__ == "__main__":
    main()
