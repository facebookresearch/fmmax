"""Tests for `examples.sorter`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp

from examples import sorter


class SorterTest(unittest.TestCase):
    def test_regression(self):
        values = sorter.optimize(steps=2, approximate_num_terms=30)
        expected_values = [3.257598, 3.249871]
        onp.testing.assert_allclose(values, expected_values, rtol=1e-4)
