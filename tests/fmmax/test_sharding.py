"""Tests for `fmmax.scattering`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import jax
import jax.numpy as jnp
import unittest
import os
from fmmax import basis, fmm, scattering

NUM_DEVICES = 8
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={NUM_DEVICES}' # Use 8 CPU devices

class ArrayShardingTest(unittest.TestCase):
    """"""

    def test_simple_sharding(use_num_devices: int):
        """"""
        