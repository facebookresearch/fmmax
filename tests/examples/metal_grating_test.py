"""Tests for `examples.metal_grating`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp
import parameterized

from examples import metal_grating
from fmmax import basis, fmm


class MetalGratingTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                fmm.Formulation.FFT,
                (-0.871915 + 0.237970j),
                (0.832493 - 0.094269j),
            ),
            (
                fmm.Formulation.JONES_DIRECT,
                (-0.871906 + 0.237973j),
                (0.901329 + 0.167032j),
            ),
        ]
    )
    def test_regression(self, formulation, expected_te, expected_tm):
        # Checks that results match reference values. This helps protect against
        # regressions in accuracy of the simulator.
        (results,) = metal_grating.convergence_study(
            approximate_num_terms=(20,),
            truncations=(basis.Truncation.CIRCULAR,),
            fmm_formulations=(formulation,),
        )
        _, _, _, rte, rtm = results
        onp.testing.assert_allclose(rte, expected_te, rtol=1e-3)
        onp.testing.assert_allclose(rtm, expected_tm, rtol=1e-3)
