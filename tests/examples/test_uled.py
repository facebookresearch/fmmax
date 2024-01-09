"""Tests for `examples.uled`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp

from examples import uled
from fmmax import basis, fmm

SIM_CONFIG_KWARGS = {
    "formulation": fmm.Formulation.POL,
    "truncation": basis.Truncation.CIRCULAR,
    "approximate_num_terms": 400,
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
                [0.495602, 0.495602, 0.227267],
                atol=1e-3,
            )

        with self.subTest("efields"):
            self.assertSequenceEqual(efields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(efields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [62.21017, 8.363873, 2.796478],
                    [8.363873, 62.21017, 2.796478],
                    [14.590969, 14.590969, 9.568378],
                ],
                rtol=1e-3,
            )

        with self.subTest("hfields"):
            self.assertSequenceEqual(hfields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(hfields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [34.4182, 393.426243, 37.876831],
                    [393.426243, 34.4182, 37.876831],
                    [142.142214, 142.142214, 1.321144],
                ],
                rtol=1e-3,
            )
