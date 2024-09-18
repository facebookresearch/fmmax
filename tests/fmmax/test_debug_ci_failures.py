"""Tests related to anisotropic and magnetic materials.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fmm, scattering, utils

# Enable 64-bit precision for higher-accuracy.
jax.config.update("jax_enable_x64", True)


class ReproTest(unittest.TestCase):

    @parameterized.parameterized.expand(((0,), (1,)))
    def test_repro(self, dummy_var):
        del dummy_var
        result = jax_calculation()
        self.assertTrue(result)


def jax_calculation():
    # Computes the TE- and TM-reflection from a metallic grating.
    formulation = fmm.Formulation.FFT
    grating_angle = 0.0
    permittivity_ambient = jnp.asarray([[1.0 + 0.0j]])
    permittivity_passivation = jnp.asarray([[2.25 + 0.0j]])
    permittivity_metal = jnp.asarray([[-7.632 + 0.731j]])

    # Permittivity of the grating layer.
    x, y = jnp.meshgrid(jnp.linspace(-0.5, 0.5), jnp.linspace(-0.5, 0.5), indexing="ij")
    position = x * jnp.cos(grating_angle) + y * jnp.sin(grating_angle)
    density = (-0.2 < position) & (position < 0.2)
    permittivity_grating = utils.interpolate_permittivity(
        permittivity_solid=permittivity_metal,
        permittivity_void=permittivity_passivation,
        density=density,
    )

    # Thickensses of ambient, passivation, grating, and metal substrate.
    thicknesses = [
        jnp.asarray(0.0),
        jnp.asarray(0.02),
        jnp.asarray(0.06),
        jnp.asarray(0.0),
    ]

    primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=220,
        truncation=basis.Truncation.CIRCULAR,
    )
    eigensolve_kwargs = {
        "wavelength": jnp.asarray(0.63),
        "in_plane_wavevector": jnp.zeros((2,)),
        "primitive_lattice_vectors": primitive_lattice_vectors,
        "expansion": expansion,
        "formulation": formulation,
    }
    solve_result_ambient = fmm.eigensolve_isotropic_media(
        permittivity=permittivity_ambient, **eigensolve_kwargs
    )
    solve_result_passivation = fmm.eigensolve_isotropic_media(
        permittivity=permittivity_passivation, **eigensolve_kwargs
    )
    solve_result_metal = fmm.eigensolve_isotropic_media(
        permittivity=permittivity_metal, **eigensolve_kwargs
    )

    # Perform the isotropic grating eigensolve and compute the zeroth-order reflectivity.
    solve_result_grating_isotropic = fmm.eigensolve_isotropic_media(
        permittivity=permittivity_grating, **eigensolve_kwargs
    )
    s_matrix_isotropic = scattering.stack_s_matrix(
        layer_solve_results=[
            solve_result_ambient,
            solve_result_passivation,
            solve_result_grating_isotropic,
            solve_result_metal,
        ],
        layer_thicknesses=thicknesses,
    )
    n = expansion.num_terms
    r_te_isotropic = s_matrix_isotropic.s21[0, 0]
    r_tm_isotropic = s_matrix_isotropic.s21[n, n]

    # Perform the anisotropic grating eigensolve and compute the zeroth-order reflectivity.
    solve_result_grating_anisotropic = fmm.eigensolve_general_anisotropic_media(
        permittivity_xx=permittivity_grating,
        permittivity_xy=jnp.zeros_like(permittivity_grating),
        permittivity_yx=jnp.zeros_like(permittivity_grating),
        permittivity_yy=permittivity_grating,
        permittivity_zz=permittivity_grating,
        permeability_xx=jnp.ones_like(permittivity_grating),
        permeability_xy=jnp.zeros_like(permittivity_grating),
        permeability_yx=jnp.zeros_like(permittivity_grating),
        permeability_yy=jnp.ones_like(permittivity_grating),
        permeability_zz=jnp.ones_like(permittivity_grating),
        **eigensolve_kwargs,
    )
    s_matrix_anisotropic = scattering.stack_s_matrix(
        layer_solve_results=[
            solve_result_ambient,
            solve_result_passivation,
            solve_result_grating_anisotropic,  # Use results of anisotropic eigensolve.
            solve_result_metal,
        ],
        layer_thicknesses=thicknesses,
    )
    r_te_anisotropic = s_matrix_anisotropic.s21[0, 0]
    r_tm_anisotropic = s_matrix_anisotropic.s21[n, n]

    return True
