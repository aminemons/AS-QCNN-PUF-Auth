# AS-QCNN-PUF-Auth

**FPGA-Accelerated Quantum Convolutional Neural Network for Nanoelectronic PUF Authentication**

Authors: Amine Allab (Phase 1 — Quantum Design) | Dhiaa Mohammed Kessad (Phase 2 — FPGA HLS)

---

## Project Structure

```
phase1_quantum/     # Python training pipeline (this repo)
phase2_fpga/        # HLS C++ hardware implementation (Dhiaa)
results/matrices/   # Exported unitary matrices → FPGA handoff
results/plots/      # Training curves, ROC, circuit diagrams
results/metrics/    # JSON metrics per PUF type
```

## Architecture

```
[64-bit PUF Challenge]
        ↓
[Classical Encoder]  Linear(64→128→64→8) + ReLU + BN
        ↓
[Quantum VQC]        8 qubits | 6 StronglyEntangling layers | AngleEmbedding
                     Pure PyTorch GPU implementation (no cuQuantum needed)
        ↓
[Classical Head]     Linear(8→1) + Sigmoid
        ↓
[Binary Response]    0 or 1
```

**PUF Types trained on:**
| PUF | Bits | CRPs | Classical ML Acc | QCNN Target |
|---|---|---|---|---|
| 1-XOR Arbiter | 64 | 1M | ~99% | ~99% |
| 3-XOR Arbiter | 64 | 1M | ~85% | ~92% |
| 5-XOR Arbiter | 64 | 1M | ~65% | ~78% |

The **5-XOR gap** demonstrates quantum advantage.

---

## Quickstart (SSH Workstation)

```bash
# 1. Clone
git clone https://github.com/aminemons/AS-QCNN-PUF-Auth.git
cd AS-QCNN-PUF-Auth

# 2. Environment
conda env create -f environment.yml
conda activate as-qcnn-puf

# 3. Generate datasets
python phase1_quantum/01_generate_puf_dataset.py --all --n_crps 1000000

# 4. Train (full run, ~3-4h on RTX A5000)
python phase1_quantum/02_train_qcnn.py --config phase1_quantum/configs/full_config.yaml

# 5. Evaluate and export matrices
python phase1_quantum/03_evaluate_and_export.py --auto

# 6. Visualize
python phase1_quantum/04_visualize_results.py --auto

# 7. Push results
bash push_results.sh
```

## Hardware Requirements

- GPU: NVIDIA RTX A5000 (24 GB VRAM) — tested
- CUDA: 12.x
- RAM: 32 GB recommended
- Storage: 20 GB free (datasets + checkpoints)

No cuQuantum required. Quantum simulation runs natively in PyTorch on CUDA.

---

## FPGA Handoff

After training, `results/matrices/` contains:
- `gate_matrices.json` — all unitary matrices in complex float
- `gate_matrices_fixed16.json` — quantized to ap_fixed<16,4>
- `hls_constants.cpp` — ready to paste into Xilinx Vitis HLS

## Citation

Based on the IEEE conference paper:
*"An FPGA-Accelerated Implementation Framework for Quantum Convolutional Neural Networks in Nanoelectronic PUF Authentication"*
Amine Allab, Dhiaa Mohammed Kessad — Micro and Nanoelectronics, Algiers
