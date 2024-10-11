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
        permittivity=jnp.ones((50, 50)), **eigensolve_kwargs
    )
    solve_result_grating_anisotropic = fmm.eigensolve_general_anisotropic_media(
        permittivity_xx=jnp.ones((50, 50)),
        permittivity_xy=jnp.zeros((50, 50)),
        permittivity_yx=jnp.zeros((50, 50)),
        permittivity_yy=jnp.ones((50, 50)),
        permittivity_zz=jnp.ones((50, 50)),
        permeability_xx=jnp.ones((50, 50)),
        permeability_xy=jnp.zeros((50, 50)),
        permeability_yx=jnp.zeros((50, 50)),
        permeability_yy=jnp.ones((50, 50)),
        permeability_zz=jnp.ones((50, 50)),
        **eigensolve_kwargs,
    )


class DebugTest(unittest.TestCase):
    def test_simple(self):
        for _ in range(5):
            jax_calculation()
