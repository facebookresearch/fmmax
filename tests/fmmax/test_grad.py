"""Tests for various jax grad functions.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest
from typing import Any, Dict

import jax
import numpy as onp
from jax import numpy as jnp
from parameterized import parameterized

from fmmax import basis, fmm, scattering

Params = Dict[str, Any]
Aux = Dict[str, Any]


def simulate_slab(
    permittivity_array,
    thickness,
    wavelength,
    in_plane_wavevector,
    primitive_lattice_vectors,
    expansion,
):
    """Simulates a slab in vacuum."""
    permittivities = [
        jnp.ones((1, 1), dtype=complex),
        permittivity_array,
        jnp.ones((1, 1), dtype=complex),
    ]

    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        for p in permittivities
    ]
    layer_thicknesses = [jnp.zeros(()), jnp.asarray(thickness), jnp.zeros(())]

    s_matrix = scattering.stack_s_matrix(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=layer_thicknesses,
    )

    r_te = s_matrix.s21[..., 0, 0]
    r_tm = s_matrix.s21[..., expansion.num_terms, expansion.num_terms]
    return jnp.abs(jnp.stack([r_te, r_tm], axis=-1))


class JaxGradTest(unittest.TestCase):
    def test_jacrev_of_all_quantities(self):
        # Checks that jacrev with respect to all simulation parameters can be computed.
        in_plane_wavevector = jnp.asarray([0.0, 0.0])
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )

        permittivity_array = (
            jax.random.uniform(jax.random.PRNGKey(0), (20, 20)) + 1 + 0j
        ) * 5
        thickness = jnp.asarray(1.3)
        wavelength = jnp.asarray([0.43, 0.45])

        reflection = simulate_slab(
            permittivity_array=permittivity_array,
            thickness=thickness,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        output_shape = reflection.shape
        self.assertSequenceEqual(output_shape, wavelength.shape + (2,))

        (
            grad_permittivity,
            grad_thickness,
            grad_wavelength,
            grad_in_plane_wavevector,
            grad_primitive_lattice_vectors,
        ) = jax.jacrev(simulate_slab, argnums=(0, 1, 2, 3, 4))(
            permittivity_array,
            thickness,
            wavelength,
            in_plane_wavevector,
            primitive_lattice_vectors,
            expansion,
        )

        # Check that gradients have the expected shape.
        self.assertSequenceEqual(
            grad_permittivity.shape,
            output_shape + permittivity_array.shape,
        )
        self.assertSequenceEqual(
            grad_thickness.shape,
            output_shape + thickness.shape,
        )
        self.assertSequenceEqual(
            grad_wavelength.shape,
            output_shape + wavelength.shape,
        )
        self.assertSequenceEqual(
            grad_in_plane_wavevector.shape,
            output_shape + in_plane_wavevector.shape,
        )
        self.assertSequenceEqual(
            grad_primitive_lattice_vectors.u.shape,
            output_shape + primitive_lattice_vectors.u.shape,
        )
        self.assertSequenceEqual(
            grad_primitive_lattice_vectors.v.shape,
            output_shape + primitive_lattice_vectors.v.shape,
        )

        # Check that gradients are nonzero.
        self.assertFalse(onp.allclose(grad_permittivity, 0.0))
        self.assertFalse(onp.allclose(grad_thickness, 0.0))
        self.assertFalse(onp.allclose(grad_wavelength, 0.0))
        self.assertFalse(onp.allclose(grad_in_plane_wavevector, 0.0))
        self.assertFalse(onp.allclose(grad_primitive_lattice_vectors.u, 0.0))
        self.assertFalse(onp.allclose(grad_primitive_lattice_vectors.v, 0.0))

    @parameterized.expand([[jax.grad], [jax.jacrev]])
    def test_jax_grad_functions(self, grad_fn):
        # Checks that gradient of a scalar loss can be computed by various means.s
        in_plane_wavevector = jnp.asarray([0.0, 0.0])
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )

        permittivity_array = jnp.ones((20, 20), dtype=complex)
        thickness = jnp.asarray(1.3)
        wavelength = jnp.asarray([0.43, 0.45])

        def loss_fn(
            permittivity_array,
            thickness,
            wavelength,
            in_plane_wavevector,
            primitive_lattice_vectors,
            expansion,
        ):
            reflection = simulate_slab(
                permittivity_array=permittivity_array,
                thickness=thickness,
                wavelength=wavelength,
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=primitive_lattice_vectors,
                expansion=expansion,
            )
            return jnp.sum(reflection)

        (
            grad_permittivity,
            grad_thickness,
            grad_wavelength,
            grad_in_plane_wavevector,
            grad_primitive_lattice_vectors,
        ) = grad_fn(loss_fn, argnums=(0, 1, 2, 3, 4))(
            permittivity_array,
            thickness,
            wavelength,
            in_plane_wavevector,
            primitive_lattice_vectors,
            expansion,
        )

        # Check that gradients have the expected shape.
        self.assertSequenceEqual(
            grad_permittivity.shape,
            permittivity_array.shape,
        )
        self.assertSequenceEqual(
            grad_thickness.shape,
            thickness.shape,
        )
        self.assertSequenceEqual(
            grad_wavelength.shape,
            wavelength.shape,
        )
        self.assertSequenceEqual(
            grad_in_plane_wavevector.shape,
            in_plane_wavevector.shape,
        )
        self.assertSequenceEqual(
            grad_primitive_lattice_vectors.u.shape,
            primitive_lattice_vectors.u.shape,
        )
        self.assertSequenceEqual(
            grad_primitive_lattice_vectors.v.shape,
            primitive_lattice_vectors.v.shape,
        )
