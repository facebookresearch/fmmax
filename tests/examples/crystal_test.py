"""Tests for `examples.ar_coating`.

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

        onp.testing.assert_allclose(onp.sum(onp.abs(ex)), 17086.355, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(ey)), 8783.0220, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(ez)), 8502.5100, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(hx)), 7810.8830, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(hy)), 14254.221, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(hz)), 19358.705, rtol=1e-4)


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

        print(onp.sum(onp.abs(hx)))
        print(onp.sum(onp.abs(hz)))

        onp.testing.assert_allclose(onp.sum(onp.abs(ex)), 23252.148, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(ey)), 0.004013, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(ez)), 6593.631, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(hx)), 0.0051368796, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(hy)), 25672.271, rtol=1e-4)
        onp.testing.assert_allclose(onp.sum(onp.abs(hz)), 0.0045878584, rtol=1e-4)
