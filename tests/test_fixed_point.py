import re
import unittest

import numpy as np

from phase1_quantum.export.fixed_point import (
    MAX_VAL, MIN_VAL, SCALE, matrix_to_hls_cpp, to_fixed_point,
)


class HlsLiteralTests(unittest.TestCase):
    def test_nonfinite_values_cannot_become_valid_hardware_constants(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                to_fixed_point(value)

    def test_literals_retain_lsb_when_converted_with_truncation(self):
        matrix = np.array([[1 / SCALE, -1 / SCALE], [MAX_VAL, MIN_VAL]], dtype=complex)
        source = matrix_to_hls_cpp(matrix, "unitary")
        literals = [float(x) for x in re.findall(r"(-?\d+\.\d+)f", source)]
        expected = []
        for value in matrix.flat:
            expected.extend((to_fixed_point(value.real), to_fixed_point(value.imag)))
        self.assertEqual([int(value * SCALE) for value in literals], expected)


if __name__ == "__main__":
    unittest.main()
