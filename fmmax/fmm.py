"""Functions related to Fourier factorization in the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import enum
import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from fmmax import basis, utils, vector


@enum.unique
class Formulation(enum.Enum):
    """Enumerates supported Fourier modal method formulations."""

    FFT: str = "fft"
    JONES_DIRECT: str = vector.JONES_DIRECT
    JONES: str = vector.JONES
    NORMAL: str = vector.NORMAL
    POL: str = vector.POL


# -----------------------------------------------------------------------------
# Functions for computing the Fourier convolution matrices.
# -----------------------------------------------------------------------------


def fourier_matrices_patterned_isotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the Fourier convolution matrices for patterned isotropic media.

    All matrices are forms of the Fourier convolution matrices defined in equation
    8 of the S4 reference; as in the reference, we assume the z-axis is separable.
    The matrix for the zz-component is returned directly, while the in-plane
    components are blocked into a single matrix.

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        eta_matrix: The Fourier convolutio matrix for the inverse of the z-component
            of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix from
            equation 15 of [2012 Liu], computed in the manner prescribed by
            `fmm_formulation`.
    """
    if formulation == Formulation.FFT:
        transverse_permittivity_matrix = _transverse_permittivity_fft(
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=formulation,
        )
    else:
        transverse_permittivity_matrix = _transverse_permittivity_vector(
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=formulation,
        )

    transform = functools.partial(
        fourier_convolution_matrix,
        expansion=expansion,
    )

    eta_matrix = transform(1 / permittivity)
    z_permittivity_matrix = transform(permittivity)
    return eta_matrix, z_permittivity_matrix, transverse_permittivity_matrix


def fourier_matrices_patterned_anisotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    permittivity_zz: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the Fourier convolution matrices for patterned isotropic media.

    All matrices are forms of the Fourier convolution matrices defined in equation
    8 of the S4 reference; as in the reference, we assume the z-axis is separable.
    The matrix for the zz-component is returned directly, while the in-plane
    components are blocked into a single matrix.

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape `(..., nx, ny)`.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        permittivity_zz: The zz-component of the permittivity tensor.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        eta_matrix: The Fourier convolutio matrix for the inverse of the z-component
            of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix from
            equation 15 of [2012 Liu], computed in the manner prescribed by
            `fmm_formulation`.
    """
    del primitive_lattice_vectors
    if formulation != Formulation.FFT:
        raise ValueError(f"Only `Formulation.FFT` is supported, but got {formulation}.")

    transform = functools.partial(
        fourier_convolution_matrix,
        expansion=expansion,
    )

    transverse_permittivity_matrix = jnp.block(
        [
            [transform(permittivity_xx), transform(permittivity_xy)],
            [transform(permittivity_yx), transform(permittivity_yy)],
        ]
    )
    eta_matrix = transform(1 / permittivity_zz)
    z_permittivity_matrix = transform(permittivity_zz)
    return eta_matrix, z_permittivity_matrix, transverse_permittivity_matrix


def _transverse_permittivity_fft(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> jnp.ndarray:
    """Computes the `eps` matrix from [2012 Liu] equation 15 using `fft` scheme.

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        The `eps` matrix.
    """
    del primitive_lattice_vectors
    eps_hat = fourier_convolution_matrix(permittivity, expansion)
    zeros = jnp.zeros_like(eps_hat)
    return jnp.block([[eps_hat, zeros], [zeros, eps_hat]])


def _transverse_permittivity_vector(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> jnp.ndarray:
    """Computes transverse permittivity using one of the vector field methods.

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        The `eps` matrix.
    """
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)

    eps_hat = _transform(permittivity)
    zeros = jnp.zeros_like(eps_hat)
    eps_matrix = jnp.block([[eps_hat, zeros], [zeros, eps_hat]])

    eta = 1 / permittivity
    eta_hat = _transform(eta)
    delta_hat = eps_hat - jnp.linalg.inv(eta_hat)
    delta_matrix = jnp.block([[delta_hat, zeros], [zeros, delta_hat]])

    vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
    tx, ty = vector_fn(permittivity, primitive_lattice_vectors)

    Pxx, Pxy, Pyx, Pyy = tangent_terms(tx, ty)
    p_matrix = jnp.block(
        [[_transform(Pyy), _transform(Pyx)], [_transform(Pxy), _transform(Pxx)]]
    )
    return eps_matrix - delta_matrix @ p_matrix


def tangent_terms(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the `(Pxx, Pxy, Pyx, Pyy)` from [2012 Liu] equation 50.

    Args:
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.

    Returns:
        The terms `(Pxx, Pxy, Pyx, Pyy)`.
    """
    denom = jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
    denom_safe = jnp.where(jnp.isclose(denom, 0), 1.0, denom)

    # Compute terms from equation 50 of [2012 Liu]. Note that there is an error
    # in equation 48-51 of [2012 Liu], where x- and y- subscripts are swapped for
    # the diagonal blocks. This can be shown by deriving equation 48 from 46, and
    # can also easily be seen by considering the case of a 1D gratng periodic in
    # the x-direction. In this case, the tangent vector field has a y-component
    # only (i.e. tx = 0), and we want to recover the transverse permittivity
    # given by equation 43. This is possible only by swapping the x- and y-
    # subscripts in the diagonal terms of equations 48-51.
    #
    # Note that we also investigated both orderings numerically with the 1D
    # grating example, and found improved convergence with this correction.
    Pyy = jnp.abs(tx) ** 2 / denom_safe
    Pyx = jnp.conj(tx) * ty / denom_safe
    Pxy = tx * jnp.conj(ty) / denom_safe
    Pxx = jnp.abs(ty) ** 2 / denom_safe
    return Pxx, Pxy, Pyx, Pyy


# -----------------------------------------------------------------------------
# Functions related to Fourier transforms.
# -----------------------------------------------------------------------------


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
    min_shape = _min_array_shape_for_expansion(expansion)
    if any([d < dmin for d, dmin in zip(shape[-2:], min_shape)]):
        raise ValueError(
            f"`shape` is insufficient for `expansion`, the minimum shape for the "
            f"final two axes is {min_shape} but got shape {shape}."
        )


def _min_array_shape_for_expansion(expansion: basis.Expansion) -> Tuple[int, int]:
    """Returns the minimum allowed shape for an array to be expanded."""
    return (
        int(2 * max(abs(expansion.basis_coefficients[:, 0])) + 1),
        int(2 * max(abs(expansion.basis_coefficients[:, 1])) + 1),
    )


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


jax.tree_util.register_pytree_node(
    Formulation,
    lambda x: ((), x.value),
    lambda value, _: Formulation(value),
)
