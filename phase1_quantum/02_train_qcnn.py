"""
Script 02 — Main Training Script

Trains HybridQCNN on 1-XOR, 3-XOR, 5-XOR Arbiter PUF datasets.
Two-phase: warm-up (VQC frozen) → joint end-to-end backprop.

Usage:
    # Quick smoke test (2 epochs, small data)
    python phase1_quantum/02_train_qcnn.py --config phase1_quantum/configs/small_config.yaml

    # Full run (3-4h on RTX A5000)
    python phase1_quantum/02_train_qcnn.py --config phase1_quantum/configs/full_config.yaml

    # Single PUF type
    python phase1_quantum/02_train_qcnn.py --config phase1_quantum/configs/full_config.yaml --puf_types 5xor
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_quantum.data import load_puf_dataloaders
from phase1_quantum.models import HybridQCNN
from phase1_quantum.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def hardware_report(device: torch.device):
    logger.info("=" * 60)
    logger.info("Hardware Report")
    logger.info(f"  PyTorch version : {torch.__version__}")
    logger.info(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        logger.info(f"  GPU             : {props.name}")
        logger.info(f"  VRAM            : {total_gb:.1f} GB")
        logger.info(f"  CUDA version    : {torch.version.cuda}")
    logger.info(f"  Device selected : {device}")
    logger.info("=" * 60)


def find_dataset(data_dir: Path, xor_k: int) -> Path:
    matches = list(data_dir.glob(f"puf_{xor_k}xor_*.npz"))
    if not matches:
        logger.error(f"No dataset found for {xor_k}-XOR in {data_dir}.")
        logger.error("Run: python phase1_quantum/01_generate_puf_dataset.py --all --n_crps 1000000")
        sys.exit(1)
    return sorted(matches)[-1]   # pick largest/latest


def train_puf_type(xor_k: int, cfg: dict, device: torch.device) -> dict:
    tag = f"{xor_k}xor"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training on {xor_k}-XOR Arbiter PUF")
    logger.info(f"{'='*60}")

    data_dir = Path(cfg.get("data_dir", "data"))
    npz_path = find_dataset(data_dir, xor_k)
    logger.info(f"Dataset: {npz_path}")

    train_loader, val_loader, test_loader = load_puf_dataloaders(
        str(npz_path),
        batch_size=cfg["batch_size"],
        val_split=0.1,
        test_split=0.1,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    model = HybridQCNN(
        n_bits=cfg.get("n_bits", 64),
        n_qubits=cfg.get("n_qubits", 8),
        n_layers=cfg.get("n_layers", 6),
    )
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: {param_counts}")

    trainer = Trainer(
        model=model,
        device=device,
        cfg=cfg,
        run_tag=tag,
        output_dir=cfg.get("output_dir", "results"),
    )

    t0 = time.time()
    best_ckpt, history = trainer.fit(train_loader, val_loader)
    elapsed = time.time() - t0

    # Quick test set evaluation
    model.eval()
    if best_ckpt and Path(best_ckpt).exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        logger.info(f"Loaded best checkpoint: {best_ckpt}")

    model = model.to(device)
    correct = total = 0
    with torch.no_grad():
        for challenges, responses in test_loader:
            challenges = challenges.to(device)
            responses  = responses.to(device)
            logits = model(challenges)
            preds  = (logits > 0).long()
            correct += (preds == responses.long()).sum().item()
            total   += len(responses)
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

    # Save best model for export step
    out_dir = Path(cfg.get("output_dir", "results")) / "checkpoints" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    torch.save(model, out_dir / "final_model_full.pt")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid QCNN on PUF datasets")
    parser.add_argument("--config",    type=str, required=True,         help="Path to YAML config")
    parser.add_argument("--puf_types", type=str, default="1xor,3xor,5xor",
                        help="Comma-separated list: 1xor,3xor,5xor")
    parser.add_argument("--device",    type=str, default="auto")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    hardware_report(device)

    puf_types = [x.strip() for x in args.puf_types.split(",")]
    xor_map = {"1xor": 1, "3xor": 3, "5xor": 5}

    all_results = []
    for puf_str in puf_types:
        if puf_str not in xor_map:
            logger.warning(f"Unknown PUF type: {puf_str}, skipping.")
            continue
        result = train_puf_type(xor_map[puf_str], cfg, device)
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
