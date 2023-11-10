"""Tests for `examples.anisotropic_grating`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp

from examples import anisotropic_grating


class AnisotropicGratingTest(unittest.TestCase):
    def test_regression(self):
        # Checks that results match reference values. This helps protect against
        # regressions in accuracy of the simulator.
        _, incident, reflected, transmitted = anisotropic_grating.simulate_grating(
            wavelength_nm=jnp.array([450.0, 550.0, 620.0]),
            polar_angle=jnp.array([[0.0], [0.0], [0.1], [0.1]]),
            azimuthal_angle=jnp.array([[0.0], [0.1], [0.0], [0.1]]),
        )

        # Sum over the orders, giving the per-wavelength and per-polarization values.
        incident = jnp.sum(incident, axis=-2)
        reflected = jnp.sum(reflected, axis=-2)
        transmitted = jnp.sum(transmitted, axis=-2)

        onp.testing.assert_allclose(-reflected + transmitted, incident, rtol=1e-5)
