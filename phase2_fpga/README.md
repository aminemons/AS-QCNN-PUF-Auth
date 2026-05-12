# Phase 2 — FPGA HLS Implementation (Dhiaa Mohammed Kessad)

This directory contains the Xilinx Vitis HLS C++ implementation.
It depends on the matrix artifacts produced by Phase 1.

## Input from Phase 1

After Phase 1 training completes, copy the following files here:

```
results/matrices/hls_constants_5xor.cpp   → QCNN gate constants
results/matrices/gate_matrices_fixed16_5xor.json → full matrix set
```

## Files (to be added)

- `top_level_qcnn.cpp`  — top-level HLS function with DATAFLOW pragma
- `apply_gate.cpp`      — processing element (PE) for 2x2 matrix-vector multiply
- `embedding.cpp`       — angle embedding hardware
- `measure_z.cpp`       — PauliZ expectation hardware
- `qcnn_types.h`        — ap_fixed<16,4> type definitions
- `testbench.cpp`       — C-simulation testbench
- `project.tcl`         — Vitis HLS project script

## HLS Directives

- `#pragma HLS PIPELINE II=1` on all PE functions
- `#pragma HLS DATAFLOW` on top-level
- `#pragma HLS ARRAY_PARTITION` on state vector arrays
