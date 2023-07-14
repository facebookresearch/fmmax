"""Tests for `examples.uled`."""

import unittest

import jax.numpy as jnp
import numpy as onp

from examples import uled
from fmmax import basis, fmm

SIM_CONFIG_KWARGS = {
    "fmm_configuration": fmm.FmmConfiguration(
        formulation=fmm.FmmFormulation.JONES_DIRECT,
        toeplitz_mode=fmm.ToeplitzMode.CIRCULANT,
    ),
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
                [0.464907, 0.387654, 0.223326],
                atol=1e-3,
            )

        with self.subTest("efields"):
            self.assertSequenceEqual(efields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(efields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [21.836815, 18.806988, 13.744629],
                    [20.652706, 24.860136, 13.474065],
                    [13.430139, 17.602402, 26.072725],
                ],
                rtol=1e-3,
            )

        with self.subTest("hfields"):
            self.assertSequenceEqual(hfields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(hfields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [117.24337, 145.54196, 101.089195],
                    [129.05042, 114.42584, 106.21891],
                    [85.74388, 92.85141, 49.98281],
                ],
                rtol=1e-3,
            )
