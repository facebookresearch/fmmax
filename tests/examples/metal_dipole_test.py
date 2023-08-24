"""Tests for `examples.metal_dipole`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp

from examples import metal_dipole


class MetalDipoleTest(unittest.TestCase):
    def test_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
        ) = metal_dipole.simulate_metal_dipole(
            approximate_num_terms=200,
            grid_spacing_fields=0.1,
        )

        self.assertSequenceEqual(ex.shape, x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        onp.testing.assert_allclose(
            onp.mean(onp.abs((ex, ey, ez))), 4.712292, rtol=1e-4
        )
        onp.testing.assert_allclose(
            onp.mean(onp.abs((hx, hy, hz))), 7.276852, rtol=1e-4
        )
