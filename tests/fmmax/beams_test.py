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
        fields = beams.rotated_fields(
            plane_wave_fields,
            x,
            y,
            z,
            polar_angle,
            azimuthal_angle,
            polarization_angle,
        )
        with self.subTest("ex"):
            onp.testing.assert_allclose(fields[0][0], expected_fields[0][0], atol=1e-5)
        with self.subTest("ey"):
            onp.testing.assert_allclose(fields[0][1], expected_fields[0][1], atol=1e-5)
        with self.subTest("ez"):
            onp.testing.assert_allclose(fields[0][2], expected_fields[0][2], atol=1e-5)
        with self.subTest("hx"):
            onp.testing.assert_allclose(fields[1][0], expected_fields[1][0], atol=1e-5)
        with self.subTest("hy"):
            onp.testing.assert_allclose(fields[1][1], expected_fields[1][1], atol=1e-5)
        with self.subTest("hz"):
            onp.testing.assert_allclose(fields[1][2], expected_fields[1][2], atol=1e-5)


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


class GaussianBeamTest(unittest.TestCase):
    def test_cross_sections(self):
        # compute the cross section of a beam propagating in different
        # directions, and ensure it remains consistent.

        wavelength = 1.0
        n = 1.0
        k = 2 * jnp.pi * n / wavelength
        beam_waist = 2.0
        beam_center = jnp.asarray([1.0, 1.0, 1.0])

        offset = 1.0
        N = 50
        start = -5.0
        stop = 5.0

        k_vector_x_p = k * jnp.asarray([1, 1, 0])
        polarization = jnp.asarray([0, 0, 1])
        x = jnp.linspace(start, stop, N)
        y = jnp.linspace(start, stop, N)
        z = jnp.asarray([offset])
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        r_pts = jnp.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        xEx, xEy, xEz, xHx, xHy, xHz = beams.get_gaussianbeam_EH(
            r_pts, k_vector_x_p, beam_waist, beam_center, polarization
        )

        k_vector_y_p = k * jnp.asarray([0, 1, 1])
        polarization = jnp.asarray([1, 0, 0])
        x = jnp.asarray([offset])
        y = jnp.linspace(start, stop, N)
        z = jnp.linspace(start, stop, N)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        r_pts = jnp.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        yEx, yEy, yEz, yHx, yHy, yHz = beams.get_gaussianbeam_EH(
            r_pts, k_vector_y_p, beam_waist, beam_center, polarization
        )

        k_vector_z_p = k * jnp.asarray([1, 0, 1])
        polarization = jnp.asarray([0, 1, 0])
        x = jnp.linspace(start, stop, N)
        y = jnp.asarray([offset])
        z = jnp.linspace(start, stop, N)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        r_pts = jnp.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        zEx, zEy, zEz, zHx, zHy, zHz = beams.get_gaussianbeam_EH(
            r_pts, k_vector_z_p, beam_waist, beam_center, polarization
        )

        onp.testing.assert_allclose(xEz, yEx)
        onp.testing.assert_allclose(yEx, zEy)
        onp.testing.assert_allclose(xEz, zEy)
