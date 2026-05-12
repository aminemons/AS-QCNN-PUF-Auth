"""
Script 04 — Visualize Results

Generates publication-quality plots:
  - Training loss/accuracy curves per PUF type
  - ROC curves comparing 1-XOR vs 3-XOR vs 5-XOR
  - Quantization error histogram
  - Confusion matrices

Usage:
    python phase1_quantum/04_visualize_results.py --auto
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

COLORS = {"1xor": "#4C9BE8", "3xor": "#F5A623", "5xor": "#E84C4C"}
STYLE  = {"linewidth": 2.0, "alpha": 0.9}


def plot_training_curves(summary_path: Path, out_dir: Path):
    """Loss and accuracy curves from training summary."""
    if not summary_path.exists():
        logger.warning(f"No training summary at {summary_path}")
        return

    with open(summary_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hybrid QCNN — Training Summary", fontsize=14, fontweight="bold")

    # Accuracy bar chart
    tags  = [r["puf_type"] for r in results]
    accs  = [r["test_acc"] for r in results]
    times = [r["train_time_min"] for r in results]
    colors = [COLORS.get(t, "#888") for t in tags]

    axes[0].bar(tags, accs, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy per PUF Type")
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random baseline")
    for i, (t, a) in enumerate(zip(tags, accs)):
        axes[0].text(i, a + 0.005, f"{a:.4f}", ha="center", va="bottom", fontsize=10)
    axes[0].legend()

    # Training time bar chart
    axes[1].bar(tags, times, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_ylabel("Training Time (min)")
    axes[1].set_title("Training Time per PUF Type")
    for i, (t, tm) in enumerate(zip(tags, times)):
        axes[1].text(i, tm + 0.3, f"{tm:.1f}m", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "training_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_eval_metrics(metrics_dir: Path, out_dir: Path):
    """Bar charts of accuracy, F1, AUC per PUF type."""
    tags = ["1xor", "3xor", "5xor"]
    metric_names = ["accuracy", "f1_score", "auc_roc"]
    labels = ["Accuracy", "F1 Score", "AUC-ROC"]

    data = {}
    for tag in tags:
        path = metrics_dir / f"eval_{tag}.json"
        if path.exists():
            with open(path) as f:
                data[tag] = json.load(f)

    if not data:
        logger.warning("No evaluation metrics found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Hybrid QCNN — Evaluation Metrics", fontsize=14, fontweight="bold")

    present_tags = [t for t in tags if t in data]
    x = np.arange(len(present_tags))

    for ax_idx, (metric, label) in enumerate(zip(metric_names, labels)):
        vals = [data[t].get(metric, 0) for t in present_tags]
        colors = [COLORS.get(t, "#888") for t in present_tags]
        axes[ax_idx].bar(present_tags, vals, color=colors, edgecolor="white", linewidth=1.2)
        axes[ax_idx].set_ylim(0.5, 1.05)
        axes[ax_idx].set_title(label)
        axes[ax_idx].set_ylabel(label)
        for i, v in enumerate(vals):
            axes[ax_idx].text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "eval_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_quantization_errors(matrices_dir: Path, out_dir: Path):
    """Histogram of fixed-point quantization errors across all gate matrices."""
    all_errors = []
    labels_info = []
    for tag in ["1xor", "3xor", "5xor"]:
        path = matrices_dir / f"gate_matrices_fixed16_{tag}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for layer_key, qubits in data.items():
            for qubit_key, q in qubits.items():
                all_errors.append(q["max_error"])
                labels_info.append(f"{tag}/{layer_key}/{qubit_key}")

    if not all_errors:
        logger.warning("No fixed-point matrices found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_errors, bins=30, color="#4C9BE8", edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.axvline(np.mean(all_errors), color="#E84C4C", linestyle="--",
               linewidth=2, label=f"Mean: {np.mean(all_errors):.5f}")
    ax.axvline(2**-12, color="#F5A623", linestyle=":", linewidth=2,
               label=f"ap_fixed<16,4> resolution: {2**-12:.5f}")
    ax.set_xlabel("Max Absolute Quantization Error")
    ax.set_ylabel("Count")
    ax.set_title("Fixed-Point Quantization Errors (ap_fixed<16,4>)")
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "quantization_errors.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto",        action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results_dir  = Path(args.results_dir)
    metrics_dir  = results_dir / "metrics"
    matrices_dir = results_dir / "matrices"
    plots_dir    = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(metrics_dir / "training_summary.json", plots_dir)
    plot_eval_metrics(metrics_dir, plots_dir)
    plot_quantization_errors(matrices_dir, plots_dir)

    logger.info(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
