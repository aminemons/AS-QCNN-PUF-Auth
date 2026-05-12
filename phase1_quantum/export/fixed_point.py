"""
Fixed-point quantization utilities for FPGA handoff.

Converts float32 complex matrix entries to ap_fixed<16,4> format:
  - 1 sign bit
  - 3 integer bits  → max value ±7
  - 12 fractional bits → resolution 2^-12 ≈ 0.000244

Reference: Xilinx Vitis HLS ap_fixed<16,4>
"""

import numpy as np
import math


TOTAL_BITS = 16
INT_BITS = 4          # includes sign
FRAC_BITS = 12
SCALE = 2 ** FRAC_BITS           # 4096
MAX_VAL =  (2 ** (INT_BITS - 1)) - 2 ** (-FRAC_BITS)   # ≈ 7.999756
MIN_VAL = -(2 ** (INT_BITS - 1))                         # -8.0


def to_fixed_point(value: float) -> int:
    """Convert a float to ap_fixed<16,4> integer representation."""
    clamped = max(MIN_VAL, min(MAX_VAL, value))
    return int(round(clamped * SCALE))


def from_fixed_point(fp_int: int) -> float:
    """Convert ap_fixed<16,4> integer back to float."""
    # Handle two's complement
    if fp_int >= 2 ** (TOTAL_BITS - 1):
        fp_int -= 2 ** TOTAL_BITS
    return fp_int / SCALE


def quantize_matrix(U: np.ndarray) -> dict:
    """
    Quantize a 2x2 complex numpy matrix to fixed-point.

    Returns dict with:
      - float_matrix: original
      - fixed_ints:   integer representation
      - reconstructed: back-converted float (to measure quantization error)
      - max_error:    max absolute element-wise error
    """
    assert U.shape == (2, 2), "Only 2x2 matrices supported"
    fixed_ints = np.zeros((2, 2, 2), dtype=np.int32)   # [i, j, {real,imag}]
    reconstructed = np.zeros((2, 2), dtype=complex)

    for i in range(2):
        for j in range(2):
            r_int = to_fixed_point(U[i, j].real)
            im_int = to_fixed_point(U[i, j].imag)
            fixed_ints[i, j, 0] = r_int
            fixed_ints[i, j, 1] = im_int
            reconstructed[i, j] = complex(from_fixed_point(r_int), from_fixed_point(im_int))

    max_error = np.abs(U - reconstructed).max()
    return {
        "float_matrix": U.tolist(),
        "fixed_ints": fixed_ints.tolist(),
        "reconstructed": reconstructed.tolist(),
        "max_error": float(max_error),
    }


def matrix_to_hls_cpp(U: np.ndarray, var_name: str) -> str:
    """
    Generate C++ constant declaration for Xilinx Vitis HLS.

    Output format:
      const q_complex VAR_NAME[2][2] = {
          {{ {real_fp, imag_fp}, {real_fp, imag_fp} }},
          {{ {real_fp, imag_fp}, {real_fp, imag_fp} }}
      };
    """
    lines = [f"const q_complex {var_name}[2][2] = {{"]
    for i in range(2):
        row_parts = []
        for j in range(2):
            r = from_fixed_point(to_fixed_point(U[i, j].real))
            im = from_fixed_point(to_fixed_point(U[i, j].imag))
            row_parts.append(f"{{{r:.6f}f, {im:.6f}f}}")
        lines.append(f"    {{{', '.join(row_parts)}}},")
    lines.append("};")
    return "\n".join(lines)
