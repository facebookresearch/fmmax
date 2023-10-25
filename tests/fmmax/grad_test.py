"""Tests for various jax grad functions.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest
from typing import Any, Dict, Tuple

import jax
import numpy as onp
import parameterized
from jax import grad, jacrev
from jax import numpy as jnp
from jax import value_and_grad

from examples import sorter
from fmmax import basis, fmm

Params = Dict[str, Any]
Aux = Dict[str, Any]


class JaxGradTest(unittest.TestCase):
    @parameterized.parameterized.expand([grad, value_and_grad, jacrev])
    def test_jax_grad_functions(self, grad_func):
        psc = sorter.PolarizationSorterComponent(approximate_num_terms=100)
        params: Params = psc.init(jax.random.PRNGKey(0))
        density = params["layers"]["sorter"]["density"]

        def loss_fn(
            params: Params,
        ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Aux]]:
            response, aux = psc.response(params)
            loss = (
                jnp.sum((jnp.zeros(response.shape) - response) ** 2) / response.shape[0]
            )
            return loss, (response, aux)

        params["layers"]["sorter"]["density"] = density * 0 + 0.5
        grad_func(loss_fn, has_aux=True)(params)
