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
[16 patches of 4 bits]
        ↓
[Shared Classical Encoder]  Dense(4→32→4), BatchNorm, ReLU, Dropout, tanh
        ↓
[Shared Quantum VQC]        4 qubits | 5 layers | TensorFlow implementation
        ↓
[Concatenated expectations] 16 patches × 4 qubits = 64 features
        ↓
[Classical Head]            Dense(64→64→64→1)
        ↓
[Binary-response logit]
```

This describes the defaults in `phase1_quantum/models/hybrid_model.py` and the qubit/layer settings in `phase1_quantum/configs/full_config.yaml`. Older training/export utilities and dependency manifests may still need alignment with this model; the diagram is not a claim that every pipeline stage has been reproduced.

**PUF evaluation targets (not benchmark results):**
| PUF | Bits | CRPs | Classical ML Acc | QCNN Target |
|---|---|---|---|---|
| 1-XOR Arbiter | 64 | 1M | ~99% | ~99% |
| 3-XOR Arbiter | 64 | 1M | ~85% | ~92% |
| 5-XOR Arbiter | 64 | 1M | ~65% | ~78% |

These values are project targets and reference estimates, not measured results established by committed experiment artifacts. The tracked `results/metrics/` and `results/plots/` directories currently contain placeholders. Evaluate the QCNN and matched classical baselines on the same held-out data before drawing performance conclusions. This target table does not demonstrate quantum advantage.

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

The current model uses TensorFlow/Keras; the conda environment requests `tensorflow[and-cuda]`. It does not require cuQuantum. The separate `requirements.txt` still lists legacy PyTorch dependencies and should be reconciled before using that installation path.

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
