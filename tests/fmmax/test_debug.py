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


def _eig_jax(matrix):
    """Eigendecomposition using `jax.numpy.linalg.eig`."""
    eigval, eigvec = jax.pure_callback(
        _eig_jax_cpu,
        (
            jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
        ),
        matrix.astype(complex),
        vectorized=True,
    )
    return jnp.asarray(eigval), jnp.asarray(eigvec)


# Define jax eigendecomposition that runs on CPU. Note that the compilation takes
# place at module import time. If the `jit` is inside a function, deadlocks can occur.
with jax.default_device(jax.devices("cpu")[0]):
    _eig_jax_cpu = jax.jit(jnp.linalg.eig)


class DebugTest(unittest.TestCase):
    def test_simple(self):
        for _ in range(10):
            _eig_jax(jnp.ones((440, 440)))
