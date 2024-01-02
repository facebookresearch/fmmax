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
                [0.48388, 0.47384, 0.260027],
                atol=1e-3,
            )

        with self.subTest("efields"):
            self.assertSequenceEqual(efields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(efields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [146.0726, 95.23531, 2.807361],
                    [96.071144, 140.05743, 2.833282],
                    [52.65253, 50.394245, 3.149207],
                ],
                rtol=1e-3,
            )

        with self.subTest("hfields"):
            self.assertSequenceEqual(hfields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(hfields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [443.65646, 786.21075, 18.77912],
                    [817.5605, 443.23544, 18.739767],
                    [433.22864, 421.48523, 7.723214],
                ],
                rtol=1e-3,
            )
