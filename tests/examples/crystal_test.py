"""Tests for `examples.ar_coating`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp

from examples import crystal


class CrystalDipoleTest(unittest.TestCase):
    def test_regression(self):
        # Checks that results match reference values. This helps protect against
        # regressions in accuracy of the simulator.
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

        onp.testing.assert_allclose(onp.sum(onp.abs(ex)), 13243.459, rtol=1e-5)
        onp.testing.assert_allclose(onp.sum(onp.abs(ey)), 5147.9634, rtol=1e-5)
        onp.testing.assert_allclose(onp.sum(onp.abs(ez)), 7286.5552, rtol=1e-5)
        onp.testing.assert_allclose(onp.sum(onp.abs(hx)), 4121.7627, rtol=1e-5)
        onp.testing.assert_allclose(onp.sum(onp.abs(hy)), 12085.564, rtol=1e-5)
        onp.testing.assert_allclose(onp.sum(onp.abs(hz)), 11874.113, rtol=1e-5)
