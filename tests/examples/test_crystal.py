"""Tests for `examples.crystal`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import numpy as onp

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)

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
            onp.mean(onp.abs((ex, ey, ez))), 0.768176, rtol=1e-4
        )
        onp.testing.assert_allclose(
            onp.mean(onp.abs((hx, hy, hz))), 0.466219, rtol=1e-4
        )


class CrystalGaussianBeamTest(unittest.TestCase):
    def test_CW_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_gaussian_beam(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
            wavelengths=crystal.WAVELENGTH,
        )

        self.assertSequenceEqual(ex.shape, (1,) + x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        onp.testing.assert_allclose(
            onp.mean(onp.abs((ex, ey, ez))), 0.270234, rtol=1e-4
        )
        onp.testing.assert_allclose(
            onp.mean(onp.abs((hx, hy, hz))), 0.199158, rtol=1e-4
        )

    def test_broadband_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_gaussian_beam(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
            wavelengths=crystal.MULTIPLE_WAVELENGTHS,
        )

        self.assertSequenceEqual(
            ex.shape, crystal.MULTIPLE_WAVELENGTHS.shape + x.shape + z.shape + (1,)
        )
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        wavelength_idx = 1

        onp.testing.assert_allclose(
            onp.mean(
                onp.abs(
                    (
                        ex[wavelength_idx, ...],
                        ey[wavelength_idx, ...],
                        ez[wavelength_idx, ...],
                    )
                )
            ),
            0.270234,
            rtol=1e-4,
        )
        onp.testing.assert_allclose(
            onp.mean(
                onp.abs(
                    (
                        hx[wavelength_idx, ...],
                        hy[wavelength_idx, ...],
                        hz[wavelength_idx, ...],
                    )
                )
            ),
            0.199158,
            rtol=1e-4,
        )
