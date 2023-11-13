"""Tests for `examples.ar_coating`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp

from examples import ar_coating


class ARCoatingTest(unittest.TestCase):
    def test_refractive_index_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "All refractive indices"):
            ar_coating.compute_reflection(
                refractive_indices=(jnp.asarray([[2.0]]),),
                thicknesses=(100.0),
                refractive_index_ambient=1.0,
                refractive_index_substrate=1.45,
                wavelength=jnp.asarray(450.0),
                incident_angle=jnp.asarray(0.0),
            )

    def test_compare_to_lumerical(self):
        wavelength = jnp.asarray([450.0, 520.0, 640.0])
        incident_angle = jnp.asarray(
            [0.0, 15 * jnp.pi / 180, 30 * jnp.pi / 180, 45 * jnp.pi / 180]
        )

        rte, rtm = ar_coating.compute_reflection(
            refractive_indices=(1.0, 1.45, 2.0, 1.45, 2.0, 1.45),
            thicknesses=(1000.0, 92.0, 75.0, 24.0, 28.0, 1000.0),
            refractive_index_ambient=1.0,
            refractive_index_substrate=1.45,
            # The second batch dimension is for the wavelength.
            wavelength=wavelength[jnp.newaxis, :],
            # The first batch dimension is for the incident angle.
            incident_angle=incident_angle[:, jnp.newaxis],
        )

        # Reference TE reflection coefficient values computed by Lumerical stack.
        lumerical_efield_rte = jnp.asarray(
            [
                [-0.0116697 - 0.029851j, -0.0403836 + 0.050517j, 0.0210578 - 0.037780j],
                [-0.0164556 - 0.0145856j, 0.0100469 + 0.07167j, -0.0090968 - 0.052744j],
                [-0.0365618 + 0.0335138j, 0.054515 - 0.07834j, -0.0720199 + 0.0533403j],
                [-0.0505926 - 0.12564j, 0.00661052 + 0.139193j, 0.116759 - 0.119862j],
            ]
        )
        expected_rte = -lumerical_efield_rte
        onp.testing.assert_allclose(
            jnp.abs(rte) ** 2, jnp.abs(expected_rte) ** 2, rtol=1e-4
        )
        onp.testing.assert_allclose(rte, expected_rte, rtol=1e-4)

        # Reference TM reflection coefficient values computed by Lumerical stack.
        lumerical_efield_rtm = jnp.asarray(
            [
                [-0.0116697 - 0.029851j, -0.0403836 + 0.050517j, 0.0210578 - 0.037780j],
                [-0.0178055 - 0.005578j, 0.0080347 + 0.059176j, 0.0009481 - 0.0489879j],
                [-0.0072645 - 0.003216j, 0.0235415 - 0.029488j, -0.070394 + 0.0028323j],
                [0.0306234 + 0.0026343j, -0.014264 - 0.0177914j, 0.103417 + 0.0322221j],
            ]
        )
        expected_rtm = -lumerical_efield_rtm
        onp.testing.assert_allclose(
            jnp.abs(rtm) ** 2, jnp.abs(expected_rtm) ** 2, rtol=1e-4
        )
        onp.testing.assert_allclose(rtm, expected_rtm, rtol=1e-4)
