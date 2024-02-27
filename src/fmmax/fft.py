"""Functions related transforming to and from the Fourier basis.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from fmmax import basis, utils


def fourier_convolution_matrix(
    x: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Computes the Fourier convolution matrix for `x` and `basis_coefficients`.

    The Fourier convolution matrix at location `(i, j)` gives the Fourier
    coefficient associated with the lattice vector obtained by subtracting the
    `j`th reciprocal lattice vector from the `i`th reciprocal lattice basis.
    See equation 8 from [2012 Liu].

    Args:
        x: The array for which the Fourier coefficients are sought.
        expansion: The field expansion to be used.

    Returns:
        The coefficients, with shape `(num_vectors, num_vectors)`.
    """
    _validate_shape_for_expansion(x.shape, expansion)

    x_fft = jnp.fft.fft2(x)
    x_fft /= jnp.prod(jnp.asarray(x.shape[-2:]))
    idx = _standard_toeplitz_indices(expansion)
    return x_fft[..., idx[..., 0], idx[..., 1]]


def _standard_toeplitz_indices(expansion: basis.Expansion) -> jnp.ndarray:
    """Computes the indices for a standard Toeplitz matrix for `basis_coefficients`.

    Args:
        expansion: The field expansion to be used.

    Returns:
        The indices, with shape `(num, num, 2)`.
    """
    i, j = jnp.meshgrid(
        jnp.arange(expansion.num_terms),
        jnp.arange(expansion.num_terms),
        indexing="ij",
    )
    basis_coefficients = jnp.asarray(expansion.basis_coefficients)
    idx = basis_coefficients[i, :] - basis_coefficients[j, :]
    return idx


def fft(
    x: jnp.ndarray,
    expansion: basis.Expansion,
    axes: Tuple[int, int] = (-2, -1),
) -> jnp.ndarray:
    """Returns the 2D Fourier transform of `x`.

    Args:
        x: The array to be transformed.
        expansion: The field expansion to be used.
        axes: The axes to be transformed, with default being `(-2, -1)`, the final axes.

    Returns:
        The transformed `x`.
    """
    axes: Tuple[int, int] = utils.absolute_axes(axes, x.ndim)  # type: ignore[no-redef]
    _validate_shape_for_expansion(tuple([x.shape[ax] for ax in axes]), expansion)

    x_fft = jnp.fft.fft2(x, axes=axes, norm="forward")

    leading_dims = len(x.shape[: axes[0]])
    trailing_dims = len(x.shape[axes[1] + 1 :])
    slices = (
        [slice(None)] * leading_dims
        + [expansion.basis_coefficients[:, 0], expansion.basis_coefficients[:, 1]]
        + [slice(None)] * trailing_dims
    )
    return x_fft[tuple(slices)]


def ifft(
    y: jnp.ndarray,
    expansion: basis.Expansion,
    shape: Tuple[int, int],
    axis: int = -1,
) -> jnp.ndarray:
    """Returns the 2D inverse Fourier transform of `x`.

    Args:
        y: The array to be transformed.
        expansion: The field expansion to be used.
        shape: The desired shape of the output array.
        axis: The axis containing the Fourier coefficients. Default is `-1`, the
            final axis.

    Returns:
        The inverse transformed `x`.
    """
    (axis,) = utils.absolute_axes((axis,), y.ndim)
    assert y.shape[axis] == expansion.basis_coefficients.shape[-2]
    x_shape = y.shape[:axis] + shape + y.shape[axis + 1 :]
    assert len(x_shape) == y.ndim + 1

    _validate_shape_for_expansion(shape, expansion)

    leading_dims = len(y.shape[:axis])
    trailing_dims = len(y.shape[axis + 1 :])
    slices = (
        [slice(None)] * leading_dims
        + [expansion.basis_coefficients[:, 0], expansion.basis_coefficients[:, 1]]
        + [slice(None)] * trailing_dims
    )

    x = jnp.zeros(x_shape, y.dtype)
    x = x.at[tuple(slices)].set(y)
    return jnp.fft.ifft2(x, axes=(leading_dims, leading_dims + 1), norm="forward")


def _validate_shape_for_expansion(
    shape: Tuple[int, ...],
    expansion: basis.Expansion,
) -> None:
    """Validates that the shape is sufficient for the provided expansion."""
    min_shape = min_array_shape_for_expansion(expansion)
    if any([d < dmin for d, dmin in zip(shape[-2:], min_shape)]):
        raise ValueError(
            f"`shape` is insufficient for `expansion`, the minimum shape for the "
            f"final two axes is {min_shape} but got shape {shape}."
        )


def min_array_shape_for_expansion(expansion: basis.Expansion) -> Tuple[int, int]:
    """Returns the minimum allowed shape for an array to be expanded."""
    with jax.ensure_compile_time_eval():
        return (
            int(2 * max(abs(expansion.basis_coefficients[:, 0])) + 1),
            int(2 * max(abs(expansion.basis_coefficients[:, 1])) + 1),
        )
