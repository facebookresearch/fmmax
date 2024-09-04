"""Defines several utility functions.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

# The `jeig` package offers several jax-wrapped implementations of eigendecomposition,
# some of which have performance benefits. However, since `jeig` has a dependency on
# pytorch, we make its use optional. If `jeig` is not available, we fall back on a
# pure-jax implementation of the eigendecomposition.
try:
    import jeig

    _JEIG_AVAILABLE = True
except ModuleNotFoundError:
    _JEIG_AVAILABLE = False


EIG_EPS_RELATIVE = 1e-12
EIG_EPS_MINIMUM = 1e-24


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
def eig(
    matrix: jnp.ndarray,
    eps_relative: float = EIG_EPS_RELATIVE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

    The custom vjp allows gradients with resepct to the eigenvectors, unlike the
    standard jax implementation of `eig`. We use an expression for the gradient
    given in [2019 Boeddeker] along with a regularization scheme that applies
    a Lorentzian broadening to a term containing the inverse difference of
    eigenvalues. The broadening is related to the maximum magnitude of the
    eigenvalues.

    [2019 Boeddeker] https://arxiv.org/abs/1701.00392

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps_relative: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps_relative
    return _eig(matrix)


def _eig_host_jax(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


def _eig(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jeig` if available, and `_eig_host_jax` if not."""
    if _JEIG_AVAILABLE:
        return jeig.eig(matrix)
    else:
        return _eig_host_jax(matrix)


def _eig_fwd(
    matrix: jnp.ndarray,
    eps_relative: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = _eig(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps_relative)


def _eig_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, float],
    grads: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None]:
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps_relative = res
    grad_eigenvalues, grad_eigenvectors = grads

    # The expression for gradient of matrix eigenvectors with respect to matrix values
    # contains a difference between eigenvalues in the denominator. This causes
    # numerical problems in situations where eigenvalues are degenerate or nearly
    # degenerate. This problem is addressed in a few different ways in existing rcwa
    # codes. Defining `delta = eigval_i - eigval_j`,
    #
    # - torcwa uses Lorentzian broadening of the form
    #      1 / delta -> delta.conj / (abs(delta)**2 + eps)
    #   with an eps value of 1e-10.
    #   (https://github.com/kch3782/torcwa/blob/main/torcwa/torch_eig.py#L29)
    #
    # - tf_rcwa uses an expression similar to torcwa, but with apparently different
    #   conjugation. Their value of eps is 1e-6. (We tested their implementation and
    #   found it fails some of our gradient validation tests.)
    #   (https://github.com/scolburn54/rcwa_tf/blob/master/src/tensor_utils.py#L89)
    #
    # - grcwa uses `1 / delta -> 1 / (delta + eps)` with an eps value of 1e-10.
    #   (https://github.com/weiliangjinca/grcwa/blob/master/grcwa/primitives.py#L44)
    #
    # Also, from the wider literature:
    #
    # - Liao et al. (https://arxiv.org/pdf/1903.09650 sec III.A.1) uses Lorentzian
    #   broadening with an eps value of 1e-12.
    #
    # One issue with Lorentzian broadening and a fixed eps is the independence from
    # matrix scale. While a given eps may be appropriate for a matrix, it would be
    # inappropriate for the same matrix scaled by e.g. 1e-6. (The gradient should
    # be identical, except scaled by 1 / 1e-6.)
    #
    # Therefore, we use Lorentzian broadening similar to torcwa, but with an eps
    # value that is computed in a way that considers the eigenvalue range.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    delta_eig = eigenvalues_i - eigenvalues_j
    eig_range_sq = jnp.amax(jnp.abs(delta_eig) ** 2, axis=(-2, -1), keepdims=True)
    eps = jnp.maximum(eps_relative * eig_range_sq, EIG_EPS_MINIMUM)
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
