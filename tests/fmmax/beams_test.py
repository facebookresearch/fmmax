"""Tests for `fmmax.beams`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import beams


class RotatedFieldsTest(unittest.TestCase):
    pass


class RotationMatrixTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            [(0, 0, 1), 0, 0, 0, (0, 0, 1)],
            [(0, 0, 1), jnp.pi / 2, 0, 0, (1, 0, 0)],
            [(0, 0, 1), jnp.pi / 4, 0, 0, (1 / jnp.sqrt(2), 0, 1 / jnp.sqrt(2))],
            [(0, 0, 1), jnp.pi / 2, jnp.pi / 2, 0, (0, 1, 0)],
            [(-0.1, 0, 1), jnp.pi / 2, 0, 0, (1, 0, 0.1)],
            [(-0.1, 0, 1), jnp.pi / 2, 0, jnp.pi / 2, (1, -0.1, 0.0)],
            [(-0.1, 0, 1), jnp.pi / 2, 0, jnp.pi, (1, 0.0, -0.1)],
            [(-0.1, 0, 1), jnp.pi / 2, 0, 3 * jnp.pi / 2, (1, 0.1, 0.0)],
            [(-0.1, 0, 1), jnp.pi / 2, 0, 2 * jnp.pi, (1, 0.0, 0.1)],
        ]
    )
    def test_rotation_matches_expected(
        self, coords, polar_angle, azimuthal_angle, polarization_angle, expected_rotated_coords
    ):
        matrix = beams.rotation_matrix(polar_angle, azimuthal_angle, polarization_angle)
        rotated_coords = matrix @ jnp.asarray(coords)[:, jnp.newaxis]
        rotated_coords = jnp.squeeze(rotated_coords, axis=-1)
        onp.testing.assert_allclose(rotated_coords, jnp.asarray(expected_rotated_coords), atol=1e-7)
