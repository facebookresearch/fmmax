"""Tests for `examples.uled`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp

from examples import uled
from fmmax import basis, fmm

SIM_CONFIG_KWARGS = {
    "formulation": fmm.Formulation.JONES_DIRECT,
    "truncation": basis.Truncation.PARALLELOGRAMIC,
    "approximate_num_terms": 100,
}


class MicroLedTest(unittest.TestCase):
    def test_regression(self):
        # Checks that results match reference values. This helps protect against
        # regressions in accuracy of the simulator.
        extraction_efficiency, _, efields, hfields, _ = uled.simulate_uled(
            resolution=25,
            resolution_fields=25,
            dipole_fwhm=0,
            dipole_y_offset=(0,),
            **SIM_CONFIG_KWARGS,
        )
        with self.subTest("extraction efficiency"):
            onp.testing.assert_allclose(
                extraction_efficiency,
                [0.661162, 0.687372, 0.310921],
                atol=1e-3,
            )

        with self.subTest("efields"):
            self.assertSequenceEqual(efields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(efields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [103.71359, 97.317444, 3.21536],
                    [98.013145, 99.4448, 3.342574],
                    [53.552944, 53.12181, 3.446113],
                ],
                rtol=1e-3,
            )

        with self.subTest("hfields"):
            self.assertSequenceEqual(hfields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(hfields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [469.103, 561.54584, 21.604614],
                    [586.9631, 470.38712, 20.628143],
                    [351.44754, 341.26596, 9.459166],
                ],
                rtol=1e-3,
            )
