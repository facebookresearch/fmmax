"""Tests for `examples.crystal`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp

from examples import crystal


class CrystalDipoleTest(unittest.TestCase):
    def test_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_internal_source(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
        )

        self.assertSequenceEqual(ex.shape, x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        onp.testing.assert_allclose(
            onp.mean(onp.abs((ex, ey, ez))), 0.466922, rtol=1e-4
        )
        onp.testing.assert_allclose(
            onp.mean(onp.abs((hx, hy, hz))), 0.306188, rtol=1e-4
        )


class CrystalGaussianBeamTest(unittest.TestCase):
    def test_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_gaussian_beam(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
        )

        self.assertSequenceEqual(ex.shape, x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        onp.testing.assert_allclose(
            onp.mean(onp.abs((ex, ey, ez))), 0.227245, rtol=1e-4
        )
        onp.testing.assert_allclose(onp.mean(onp.abs((hx, hy, hz))), 0.17809, rtol=1e-4)
