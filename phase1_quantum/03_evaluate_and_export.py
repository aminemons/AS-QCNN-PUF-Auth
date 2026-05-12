"""
Script 03 — Evaluate and Export

Loads trained checkpoints, computes full metrics (accuracy, F1, AUC-ROC,
PUF uniqueness, PUF reliability), and exports unitary matrices for FPGA.

Usage:
    python phase1_quantum/03_evaluate_and_export.py --auto
    python phase1_quantum/03_evaluate_and_export.py --ckpt_dir results/checkpoints/5xor --tag 5xor
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_quantum.models import HybridQCNN
from phase1_quantum.data import load_puf_dataloaders
from phase1_quantum.export import MatrixExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device) -> dict:
    model.eval()
    all_logits = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for challenges, responses in test_loader:
            challenges = challenges.to(device)
            logits = model(challenges)
            preds = (logits > 0).long()
            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(responses.long().cpu())

    logits = torch.cat(all_logits).numpy()
    preds  = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))   # sigmoid

    metrics = {
        "accuracy":  float(accuracy_score(targets, preds)),
        "f1_score":  float(f1_score(targets, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(targets, probs)),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
    }
    return metrics, preds, targets


def compute_puf_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute PUF-specific metrics."""
    # Uniqueness: fraction of correct predictions (authentication accuracy)
    # Reliability: consistency — how often does pred match target
    n = len(preds)
    reliability = float((preds == targets).mean())
    # Bit-aliasing: fraction of 1s in responses (ideal 0.5)
    bit_aliasing = float(targets.mean())
    return {
        "reliability": reliability,
        "bit_aliasing": bit_aliasing,
        "ideal_deviation": abs(bit_aliasing - 0.5),
    }


def run_evaluation(tag: str, ckpt_path: Path, npz_path: Path, cfg: dict, device: torch.device):
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {tag}")
    logger.info(f"{'='*60}")

    model = HybridQCNN(
        n_bits=cfg.get("n_bits", 64),
        n_qubits=cfg.get("n_qubits", 8),
        n_layers=cfg.get("n_layers", 6),
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    _, _, test_loader = load_puf_dataloaders(
        str(npz_path),
        batch_size=cfg.get("batch_size", 4096),
        num_workers=4,
    )

    metrics, preds, targets = evaluate_model(model, test_loader, device)
    puf_metrics = compute_puf_metrics(preds, targets)
    metrics.update(puf_metrics)

    logger.info(f"Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score : {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC  : {metrics['auc_roc']:.4f}")
    logger.info(f"Reliability: {metrics['reliability']:.4f}")
    logger.info(f"Bit aliasing: {metrics['bit_aliasing']:.4f}")

    # Save metrics
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"eval_{tag}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    # Export matrices
    exporter = MatrixExporter(output_dir="results/matrices")
    export_result = exporter.export(model, tag=tag)
    logger.info(f"Quantization mean error: {export_result['mean_error']:.6f}")
    logger.info(f"Quantization max error:  {export_result['max_error']:.6f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto",       action="store_true",       help="Auto-detect all trained checkpoints")
    parser.add_argument("--ckpt_dir",   type=str, default=None)
    parser.add_argument("--tag",        type=str, default=None)
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_qubits",   type=int, default=8)
    parser.add_argument("--n_layers",   type=int, default=6)
    parser.add_argument("--n_bits",     type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"n_bits": args.n_bits, "n_qubits": args.n_qubits,
           "n_layers": args.n_layers, "batch_size": args.batch_size}

    data_dir = Path(args.data_dir)
    all_results = {}

    if args.auto:
        ckpt_base = Path("results/checkpoints")
        for tag in ["1xor", "3xor", "5xor"]:
            ckpt_path = ckpt_base / tag / "final_model.pt"
            if not ckpt_path.exists():
                logger.warning(f"No checkpoint found for {tag}: {ckpt_path}")
                continue
            npz_matches = list(data_dir.glob(f"puf_{tag.replace('xor','')}_*.npz"))
            if not npz_matches:
                logger.warning(f"No dataset found for {tag}")
                continue
            npz_path = sorted(npz_matches)[-1]
            xor_k = int(tag.replace("xor", ""))
            npz_path = sorted(data_dir.glob(f"puf_{xor_k}xor_*.npz"))[-1]
            result = run_evaluation(tag, ckpt_path, npz_path, cfg, device)
            all_results[tag] = result
    else:
        if not args.ckpt_dir or not args.tag:
            parser.error("Provide --ckpt_dir and --tag, or use --auto")
        ckpt_path = Path(args.ckpt_dir) / "final_model.pt"
        xor_k = int(args.tag.replace("xor", ""))
        npz_matches = list(data_dir.glob(f"puf_{xor_k}xor_*.npz"))
        if not npz_matches:
            logger.error(f"No dataset for {args.tag} in {data_dir}")
            sys.exit(1)
        npz_path = sorted(npz_matches)[-1]
        result = run_evaluation(args.tag, ckpt_path, npz_path, cfg, device)
        all_results[args.tag] = result

    # Combined summary
    summary_path = Path("results/metrics/eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nFull evaluation summary: {summary_path}")
    logger.info("\nFinal Results:")
    for tag, m in all_results.items():
        logger.info(f"  {tag:8s} → acc={m['accuracy']:.4f}  f1={m['f1_score']:.4f}  auc={m['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
