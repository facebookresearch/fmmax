"""Tests for `examples.metal_pillars`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)

from examples import metal_pillars
from fmmax import fmm


class MetalPillarsTest(unittest.TestCase):
    def test_regression(self):
        # Checks that results match reference values. This helps protect against
        # regressions in accuracy of the simulator.
        n, rte, _, _ = metal_pillars.simulate_pillars(
            wavelength_nm=jnp.array([450.0]),
            approximate_num_terms=400,
            ambient_thickness_nm=0.0,
            formulation=fmm.Formulation.NORMAL,
        )
        onp.testing.assert_allclose(rte, [0.069906 - 0.050099j], rtol=1e-2)

    def test_compute_fields_regression(self):
        (
            _,
            _,
            _,
            (layer_solve_results, thicknesses, s_matrices_interior),
        ) = metal_pillars.simulate_pillars(
            wavelength_nm=jnp.array([450.0, 550.0]),
            approximate_num_terms=100,
            formulation=fmm.Formulation.NORMAL,
        )
        efields, hfields = metal_pillars.compute_fields(
            layer_solve_results=layer_solve_results,
            thicknesses=thicknesses,
            s_matrices_interior=s_matrices_interior,
            resolution_nm=5.0,
        )
        self.assertSequenceEqual(efields.shape, (3, 2, 36, 36, 240, 1))
        self.assertSequenceEqual(hfields.shape, (3, 2, 36, 36, 240, 1))

        with self.subTest("efields"):
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(efields) ** 2, axis=(2, 3, 4, 5)),
                onp.asarray(
                    [[0.046316, 0.052124], [1.047591, 1.859426], [0.145127, 0.077935]]
                ),
                rtol=0.01,
            )
        with self.subTest("hfields"):
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(hfields) ** 2, axis=(2, 3, 4, 5)),
                onp.asarray(
                    [[1.037612, 1.810816], [0.004971, 0.003377], [0.045081, 0.048233]]
                ),
                rtol=0.01,
            )
