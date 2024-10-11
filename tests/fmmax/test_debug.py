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


def jax_calculation():
    # Computes the TE- and TM-reflection from a metallic grating.
    permittivity_passivation = jnp.asarray([[1.0]])
    permittivity_metal = jnp.asarray([[1.0]])

    # Permittivity of the grating layer.
    permittivity_grating = utils.interpolate_permittivity(
        permittivity_solid=permittivity_metal,
        permittivity_void=permittivity_passivation,
        density=jnp.ones((50, 50)),
    )

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
        "formulation": fmm.Formulation.JONES_DIRECT,
    }

    solve_result_grating_isotropic = fmm.eigensolve_isotropic_media(
        permittivity=permittivity_grating, **eigensolve_kwargs
    )
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


class DebugTest(unittest.TestCase):
    def test_simple(self):
        for _ in range(5):
            jax_calculation()
