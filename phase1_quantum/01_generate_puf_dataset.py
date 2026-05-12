"""
Script 01 — PUF Dataset Generator

Generates Arbiter PUF Challenge-Response Pairs using pypuf.
Supports 1-XOR, 3-XOR, 5-XOR Arbiter PUF (64-bit challenges).

Usage:
    python phase1_quantum/01_generate_puf_dataset.py --all --n_crps 1000000
    python phase1_quantum/01_generate_puf_dataset.py --xor_k 3 --n_crps 500000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_one(n_bits: int, n_crps: int, xor_k: int, noise: float, seed: int, out_dir: Path) -> Path:
    try:
        from pypuf.simulation import ArbiterPUF, XORArbiterPUF
        from pypuf.io import random_inputs
    except ImportError:
        logger.error("pypuf not found. Run: pip install pypuf")
        sys.exit(1)

    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_crps:,} CRPs — {xor_k}-XOR Arbiter PUF ({n_bits} bits, noise={noise})")

    if xor_k == 1:
        puf = ArbiterPUF(n=n_bits, seed=seed, noisiness=noise)
    else:
        puf = XORArbiterPUF(n=n_bits, k=xor_k, seed=seed, noisiness=noise)

    challenges = random_inputs(n=n_bits, N=n_crps, seed=seed)
    responses = puf.eval(challenges)
    responses = ((responses + 1) // 2).astype(np.uint8)   # {-1,+1} → {0,1}

    # Sanity checks
    balance = responses.mean()
    uniqueness = len(set(map(tuple, challenges[:10000]))) / min(10000, n_crps)

    logger.info(f"Response balance: {balance:.4f} (ideal 0.500)")
    logger.info(f"Challenge uniqueness (10k sample): {uniqueness:.4f} (ideal 1.000)")
    if abs(balance - 0.5) > 0.05:
        logger.warning("Response balance deviates >5% from ideal — check PUF parameters.")

    out_path = out_dir / f"puf_{xor_k}xor_{n_bits}bit_{n_crps}crps_seed{seed}.npz"
    np.savez_compressed(str(out_path), challenges=challenges, responses=responses)
    logger.info(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="PUF CRP Dataset Generator")
    parser.add_argument("--n_bits",   type=int,   default=64,         help="Challenge bit width")
    parser.add_argument("--n_crps",   type=int,   default=1_000_000,  help="Number of CRPs")
    parser.add_argument("--xor_k",    type=int,   default=1,          choices=[1, 3, 5])
    parser.add_argument("--noise",    type=float, default=0.05,       help="Manufacturing noise sigma")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out_dir",  type=str,   default="data")
    parser.add_argument("--all",      action="store_true",             help="Generate all XOR variants")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.all:
        for k in [1, 3, 5]:
            generate_one(args.n_bits, args.n_crps, k, args.noise, args.seed + k, out_dir)
        logger.info("All datasets generated.")
    else:
        generate_one(args.n_bits, args.n_crps, args.xor_k, args.noise, args.seed, out_dir)


if __name__ == "__main__":
    main()
