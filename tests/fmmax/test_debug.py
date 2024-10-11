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


class DebugTest(unittest.TestCase):
    def test_simple(self):
        for i in range(10):
            utils.eig(jnp.ones((440, 440)))
