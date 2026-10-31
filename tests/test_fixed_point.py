import re
import unittest

import numpy as np

from phase1_quantum.export.fixed_point import (
    MAX_VAL, MIN_VAL, SCALE, matrix_to_hls_cpp, to_fixed_point, from_fixed_point,
)


class HlsLiteralTests(unittest.TestCase):
    def test_numpy_unsigned_words_decode_without_overflow(self):
        self.assertEqual(from_fixed_point(np.uint16(65535)), -1 / SCALE)
        self.assertEqual(from_fixed_point(np.uint16(32768)), MIN_VAL)
        self.assertEqual(from_fixed_point(-1), -1 / SCALE)

    def test_invalid_word_widths_and_types_are_rejected(self):
        for value in (-32769, 65536):
            with self.subTest(value=value), self.assertRaises(ValueError):
                from_fixed_point(value)
        for value in (True, 1.5):
            with self.subTest(value=value), self.assertRaises(TypeError):
                from_fixed_point(value)

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
