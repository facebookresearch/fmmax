"""Tests comparing gradients to finite difference gradients.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from scipy import ndimage

from fmmax import basis, fields, fmm, scattering


def gaussian_permittivity_fn(x0, y0, dim):
    x = jnp.arange(0, dim)[:, jnp.newaxis] / dim - x0
    y = jnp.arange(0, dim)[jnp.newaxis, :] / dim - y0
    return 1 + 10 * jnp.exp((-(x**2) - y**2) * 25)


class FiniteDifferenceGradientTest(unittest.TestCase):
    @parameterized.expand(
        [
            (fmm.Formulation.FFT, 3e-3),
            (fmm.Formulation.JONES_DIRECT, 1e-2),
            (fmm.Formulation.JONES_DIRECT_FOURIER, 1e-2),
            (fmm.Formulation.POL, 2e-2),
            (fmm.Formulation.POL_FOURIER, 2e-2),
        ]
    )
    def test_gradient_matches_expected(self, formulation, rtol):
        in_plane_wavevector = jnp.zeros((2,))
        wavelength = jnp.asarray(0.23)
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=300,
            truncation=basis.Truncation.CIRCULAR,
        )

        def compute_fom(x0, y0, formulation):
            # Generate a permittivity distribution that has a lensing effect.
            permittivity_pattern = gaussian_permittivity_fn(x0, y0, dim=200)

            eigensolve = functools.partial(
                fmm.eigensolve_isotropic_media,
                wavelength=jnp.asarray(wavelength),
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=primitive_lattice_vectors,
                expansion=expansion,
                formulation=formulation,
            )

            solve_result_ambient = eigensolve(permittivity=jnp.full((1, 1), 1.0 + 0.0j))
            solve_result_pattern = eigensolve(permittivity=permittivity_pattern)

            layer_solve_results = [
                solve_result_ambient,
                solve_result_pattern,
                solve_result_ambient,
            ]
            layer_thicknesses = [jnp.asarray(0.2), jnp.asarray(0.1), jnp.ones(())]

            s_matrices_interior = scattering.stack_s_matrices_interior(
                layer_solve_results=layer_solve_results,
                layer_thicknesses=layer_thicknesses,
            )

            fwd_flux_ambient_start = jnp.zeros(
                (2 * expansion.num_terms, 1), dtype=complex
            )
            fwd_flux_ambient_start = fwd_flux_ambient_start.at[0, 0].set(1)

            amplitudes_interior = fields.stack_amplitudes_interior(
                s_matrices_interior=s_matrices_interior,
                forward_amplitude_0_start=fwd_flux_ambient_start,
                backward_amplitude_N_end=jnp.zeros_like(fwd_flux_ambient_start),
            )

            # Compute the fields for a cross section through the center of a lens.
            (ex, ey, ez), _, (x, y, z) = fields.stack_fields_3d(
                amplitudes_interior=amplitudes_interior,
                layer_solve_results=layer_solve_results,
                layer_thicknesses=layer_thicknesses,
                layer_znum=[10, 5, 50],
                grid_shape=(50, 50),
                num_unit_cells=(1, 1),
            )

            efield_magnitude = jnp.sqrt(
                jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2 + jnp.abs(ez) ** 2
            )

            # Figure of merit is the field magnitude near the half-maximum of the intensity. Small
            # shifts in the lens will have a large impact on the field amplitude.
            fom = efield_magnitude[28, 28, 25, 0]

            return fom, (efield_magnitude, (x, y, z), permittivity_pattern)

        # Finite difference gradient.
        eps = 1e-2
        fom0, _ = compute_fom(x0=0.5 - eps / 2, y0=0.5, formulation=formulation)
        fom1, _ = compute_fom(x0=0.5 + eps / 2, y0=0.5, formulation=formulation)
        grad_fd = (fom1 - fom0) / eps

        grad, _ = jax.grad(compute_fom, has_aux=True)(0.5, 0.5, formulation)

        onp.testing.assert_allclose(grad, grad_fd, rtol=rtol)
