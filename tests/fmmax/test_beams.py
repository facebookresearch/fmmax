"""Tests for `fmmax.beams`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import beams


class RotatedFieldsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            [0, 0, 0],
            [0, 0, jnp.pi / 2],
            [jnp.pi / 4, 0, 0],
            [jnp.pi / 4, jnp.pi / 4, 0],
            [jnp.pi / 4, jnp.pi / 4, jnp.pi / 2],
            [jnp.pi / 3, jnp.pi / 3, jnp.pi / 3],
        ]
    )
    def test_rotated_plane_wave_matches_expected(
        self, polar_angle, azimuthal_angle, polarization_angle
    ):
        wavelength = 0.314

        def plane_wave_fields(x, y, z):
            # Fields for an x-polarized plane wave propagating in the z-direction.
            del x, y
            ex = jnp.exp(1j * 2 * jnp.pi / wavelength * z)
            ey = jnp.zeros_like(ex)
            ez = jnp.zeros_like(ex)
            hx = jnp.zeros_like(ex)
            hy = ex
            hz = jnp.zeros_like(ex)
            return (ex, ey, ez), (hx, hy, hz)

        def rotated_plane_wave_fields(x, y, z):
            # Fields for a plane wae propagating in the direction defined by
            # `polar_angle` and `azimuthal_angle`, with polarization rotated by
            # the specified `polarization_angle`.
            kx = (
                2
                * jnp.pi
                / wavelength
                * jnp.sin(polar_angle)
                * jnp.cos(azimuthal_angle)
            )
            ky = (
                2
                * jnp.pi
                / wavelength
                * jnp.sin(polar_angle)
                * jnp.sin(azimuthal_angle)
            )
            kz = 2 * jnp.pi / wavelength * jnp.cos(polar_angle)
            exponential = jnp.exp(1j * (kx * x + ky * y + kz * z))

            # Define the two unit vectors for the polarization axes.
            pxx = jnp.cos(polar_angle) * jnp.cos(azimuthal_angle)
            pxy = jnp.cos(polar_angle) * jnp.sin(azimuthal_angle)
            pxz = -jnp.sin(polar_angle)

            pyx = -jnp.sin(azimuthal_angle)
            pyy = jnp.cos(azimuthal_angle)
            pyz = jnp.zeros_like(azimuthal_angle)

            # Check that the unit vectors are orthogonal.
            assert onp.isclose(pxx * pyx + pxy * pyy + pxz * pyz, 0.0)

            aex = pxx * jnp.cos(polarization_angle) + pyx * jnp.sin(polarization_angle)
            aey = pxy * jnp.cos(polarization_angle) + pyy * jnp.sin(polarization_angle)
            aez = pxz * jnp.cos(polarization_angle) + pyz * jnp.sin(polarization_angle)

            ahx = -pxx * jnp.sin(polarization_angle) + pyx * jnp.cos(polarization_angle)
            ahy = -pxy * jnp.sin(polarization_angle) + pyy * jnp.cos(polarization_angle)
            ahz = -pxz * jnp.sin(polarization_angle) + pyz * jnp.cos(polarization_angle)

            ex = aex * exponential
            ey = aey * exponential
            ez = aez * exponential
            hx = ahx * exponential
            hy = ahy * exponential
            hz = ahz * exponential
            return (ex, ey, ez), (hx, hy, hz)

        x, y, z = jnp.meshgrid(
            jnp.linspace(0, 1),
            jnp.linspace(0, 2),
            jnp.linspace(-1, 1),
            indexing="ij",
        )

        expected_fields = rotated_plane_wave_fields(x, y, z)
        fields = beams.shifted_rotated_fields(
            plane_wave_fields,
            x,
            y,
            z,
            beam_origin_x=0,
            beam_origin_y=0,
            beam_origin_z=0,
            polar_angle=polar_angle,
            azimuthal_angle=azimuthal_angle,
            polarization_angle=polarization_angle,
        )
        with self.subTest("ex"):
            onp.testing.assert_allclose(fields[0][0], expected_fields[0][0], atol=2e-5)
        with self.subTest("ey"):
            onp.testing.assert_allclose(fields[0][1], expected_fields[0][1], atol=2e-5)
        with self.subTest("ez"):
            onp.testing.assert_allclose(fields[0][2], expected_fields[0][2], atol=2e-5)
        with self.subTest("hx"):
            onp.testing.assert_allclose(fields[1][0], expected_fields[1][0], atol=2e-5)
        with self.subTest("hy"):
            onp.testing.assert_allclose(fields[1][1], expected_fields[1][1], atol=2e-5)
        with self.subTest("hz"):
            onp.testing.assert_allclose(fields[1][2], expected_fields[1][2], atol=2e-5)

    @parameterized.parameterized.expand(
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ]
    )
    def test_shifts_match_expected(self, beam_origin_x, beam_origin_y, beam_origin_z):
        def field_fn(x, y, z):
            ex = jnp.exp(-(x**2) - y**2 - z**2)
            return (ex, ex, ex), (ex, ex, ex)

        def shifted_field_fn(x, y, z):
            ex = jnp.exp(
                -((x - beam_origin_x) ** 2)
                - (y - beam_origin_y) ** 2
                - (z - beam_origin_z) ** 2
            )
            return (ex, ex, ex), (ex, ex, ex)

        x, y, z = jnp.meshgrid(
            jnp.linspace(0, 1),
            jnp.linspace(0, 2),
            jnp.linspace(-1, 1),
            indexing="ij",
        )
        fields = beams.shifted_rotated_fields(
            field_fn, x, y, z, beam_origin_x, beam_origin_y, beam_origin_z, 0, 0, 0
        )
        expected = shifted_field_fn(x, y, z)
        onp.testing.assert_allclose(fields, expected)


class RotationMatrixTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # Check that rotations move the z-axis.
            [(0, 0, 1), (0, 0, 1), 0, 0, 0],
            [(0, 0, 1), (1, 0, 0), jnp.pi / 2, 0, 0],
            [(0, 0, 1), (1 / jnp.sqrt(2), 0, 1 / jnp.sqrt(2)), jnp.pi / 4, 0, 0],
            [(0, 0, 1), (0, 1, 0), jnp.pi / 2, jnp.pi / 2, 0],
            [(1, 0, 0), (0, 1, 0), 0, 0, jnp.pi / 2],
            # Check rotations about the propagation axis.
            [(-0.1, 0, 1), (1, 0, 0.1), jnp.pi / 2, 0, 0],
            [(-0.1, 0, 1), (1, -0.1, 0.0), jnp.pi / 2, 0, jnp.pi / 2],
            [(-0.1, 0, 1), (1, 0.0, -0.1), jnp.pi / 2, 0, jnp.pi],
            [(-0.1, 0, 1), (1, 0.1, 0.0), jnp.pi / 2, 0, 3 * jnp.pi / 2],
            [(-0.1, 0, 1), (1, 0.0, 0.1), jnp.pi / 2, 0, 2 * jnp.pi],
        ]
    )
    def test_rotation_matches_expected(
        self,
        coords,
        expected_rotated_coords,
        polar_angle,
        azimuthal_angle,
        polarization_angle,
    ):
        matrix = beams.rotation_matrix(polar_angle, azimuthal_angle, polarization_angle)
        rotated_coords = matrix @ jnp.asarray(coords)[:, jnp.newaxis]
        rotated_coords = jnp.squeeze(rotated_coords, axis=-1)
        onp.testing.assert_allclose(
            rotated_coords, jnp.asarray(expected_rotated_coords), atol=1e-7
        )
