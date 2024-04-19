"""Defines several utility functions.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as onp

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

EPS_EIG = 1e-6


def diag(x: jnp.ndarray) -> jnp.ndarray:
    """A batch-compatible version of `numpy.diag`."""
    shape = x.shape + (x.shape[-1],)
    y = jnp.zeros(shape, x.dtype)
    i = jnp.arange(x.shape[-1])
    return y.at[..., i, i].set(x)


def angular_frequency_for_wavelength(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Returns the angular frequency for the specified wavelength."""
    return 2 * jnp.pi / wavelength  # Since by our convention c == 1.


def matrix_adjoint(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the adjoint for a batch of matrices."""
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jnp.conj(jnp.transpose(x, axes=axes))


def batch_compatible_shapes(*shapes: Tuple[int, ...]) -> bool:
    """Returns `True` if all the shapes are batch-compatible."""
    max_dims = max([len(s) for s in shapes])
    shapes = tuple([(1,) * (max_dims - len(s)) + s for s in shapes])
    max_shape = [max(dim_shapes) for dim_shapes in zip(*shapes)]
    for shape in shapes:
        if any([a not in (1, b) for a, b in zip(shape, max_shape)]):
            return False
    return True


def atleast_nd(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Adds leading dimensions to `x`, ensuring that it is at least n-dimensional."""
    dims_to_add = tuple(range(max(0, n - x.ndim)))
    return jnp.expand_dims(x, axis=dims_to_add)


def absolute_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    """Returns the absolute axes for given relative axes and number of array dimensions."""
    if not all(a in list(range(-ndim, ndim)) for a in axes):
        raise ValueError(
            f"All elements of `axes` must be in the range ({ndim}, {ndim - 1}) "
            f"but got {axes}."
        )
    absolute_axes = tuple([d % ndim for d in axes])
    if len(absolute_axes) != len(set(absolute_axes)):
        raise ValueError(
            f"Found duplicates in `axes`; computed absolute axes are {absolute_axes}."
        )
    return absolute_axes


def interpolate_permittivity(
    permittivity_solid: jnp.ndarray,
    permittivity_void: jnp.ndarray,
    density: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolates the permittivity with a scheme that avoids zero crossings.

    The interpolation uses the scheme introduced in [2019 Christiansen], which avoids
    zero crossings that can occur with metals or lossy materials having a negative
    real component of the permittivity. https://doi.org/10.1016/j.cma.2018.08.034

    Args:
        permittivity_solid: The permittivity of solid regions.
        permittivity_void: The permittivity of void regions.
        density: The density, specifying which locations are solid and which are void.

    Returns:
        The interpolated permittivity.
    """
    n_solid = jnp.real(jnp.sqrt(permittivity_solid))
    k_solid = jnp.imag(jnp.sqrt(permittivity_solid))
    n_void = jnp.real(jnp.sqrt(permittivity_void))
    k_void = jnp.imag(jnp.sqrt(permittivity_void))
    n = density * n_solid + (1 - density) * n_void
    k = density * k_solid + (1 - density) * k_void
    return (n + 1j * k) ** 2


# -----------------------------------------------------------------------------
# Functions related to a generalized eigensolve with custom vjp rule.
# -----------------------------------------------------------------------------


@jax.custom_vjp
def eig(matrix: jnp.ndarray, eps: float = EPS_EIG) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

    The custom vjp allows gradients with resepct to the eigenvectors, unlike the
    standard jax implementation of `eig`. We use an expression for the gradient
    given in [2019 Boeddeker] along with a regularization scheme used in [2021
    Colburn]. The method effectively applies a Lorentzian broadening to a term
    containing the inverse difference of eigenvalues.

    [2019 Boeddeker] https://arxiv.org/abs/1701.00392
    [2021 Coluburn] https://www.nature.com/articles/s42005-021-00568-6

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps
    return _eig_host(matrix)


def _eig_host(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""

    def _eig_cpu(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        with jax.default_device(jax.devices("cpu")[0]):
            return jax.jit(jnp.linalg.eig)(matrix)

    return jax.pure_callback(
        _eig_cpu,
        (
            jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
        ),
        matrix.astype(complex),
        vectorized=True,
    )


def _eig_fwd(
    matrix: jnp.ndarray,
    eps: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = _eig_host(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps)


def _eig_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, float],
    grads: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None]:
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps = res
    grad_eigenvalues, grad_eigenvectors = grads

    # Compute the broadened F-matrix. The expression is similar to that of equation 5
    # from [2021 Colburn], but differs slightly from both the code and paper.
    # TODO: derive the proper expression for broadened F-matrix.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    delta_eig = eigenvalues_i - eigenvalues_j
    f_broadened = delta_eig.conj() / (jnp.abs(delta_eig) ** 2 + eps)

    # Manually set the diagonal elements to zero, as we do not use broadening here.
    i = jnp.arange(f_broadened.shape[-1])
    f_broadened = f_broadened.at[..., i, i].set(0)

    # By jax convention, gradients are with respect to the complex parameters, not with
    # respect to their conjugates. Take the conjugates.
    grad_eigenvalues_conj = jnp.conj(grad_eigenvalues)
    grad_eigenvectors_conj = jnp.conj(grad_eigenvectors)

    eigenvectors_H = matrix_adjoint(eigenvectors)
    dim = eigenvalues.shape[-1]
    eye_mask = jnp.eye(dim, dtype=bool)
    eye_mask = eye_mask.reshape((1,) * (eigenvalues.ndim - 1) + (dim, dim))

    # Then, the gradient is found by equation 4.77 of [2019 Boeddeker].
    rhs = (
        diag(grad_eigenvalues_conj)
        + jnp.conj(f_broadened) * (eigenvectors_H @ grad_eigenvectors_conj)
        - jnp.conj(f_broadened)
        * (eigenvectors_H @ eigenvectors)
        @ jnp.where(eye_mask, jnp.real(eigenvectors_H @ grad_eigenvectors_conj), 0.0)
    ) @ eigenvectors_H
    grad_matrix = jnp.linalg.solve(eigenvectors_H, rhs)

    # Take the conjugate of the gradient, reverting to the jax convention
    # where gradients are with respect to complex parameters.
    grad_matrix = jnp.conj(grad_matrix)

    # Return `grad_matrix`, and `None` for the gradient with respect to `eps`.
    return grad_matrix, None


eig.defvjp(_eig_fwd, _eig_bwd)

# -----------------------------------------------------------------------------
# Functions related to a multi-device computation (e.g. sharding)
# -----------------------------------------------------------------------------


def _shard_wrapper(
    x: jnp.ndarray, sharding: PositionalSharding, device_axis: int, num_devices: int
) -> jnp.asarray:
    """"""
    x = jnp.asarray(x)

    # We want to replicate scalars and other "non-batchable" quantities like
    if x.ndim < 3:
        return jax.device_put(x, device=sharding.replicate(axis=0, keepdims=True))

    shard_shape = [
        1,
    ] * x.ndim
    shard_shape[device_axis] = num_devices
    x = jnp.broadcast_to(x, [max(d, s) for d, s in zip(x.shape, shard_shape)])
    return jax.device_put(x, device=sharding.reshape(shard_shape))


def shard_structure(x, num_devices):
    """"""
    # Only shard if we request enough devices
    if num_devices <= 1:
        return x

    return jax.tree_util.tree_map(
        lambda _x: _shard_wrapper(_x, _sharding, _gpu_batch_axis, _num_gpus), x
    )
