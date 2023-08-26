"""Tests for `examples.microlens_array`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp

from examples import microlens_array


class MicrolensArrayTest(unittest.TestCase):
    def test_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            _,
        ) = microlens_array.simulate_microlens_array(
            approximate_num_terms=200,
            grid_spacing_fields=0.1,
            num_lens_layers=4,
        )

        self.assertSequenceEqual(ex.shape, x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        onp.testing.assert_allclose(
            onp.mean(onp.abs((ex, ey, ez))), 0.324081, rtol=1e-4
        )
        onp.testing.assert_allclose(
            onp.mean(onp.abs((hx, hy, hz))), 0.390007, rtol=1e-4
        )
