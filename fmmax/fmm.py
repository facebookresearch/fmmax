"""Functions related to Fourier factorization in the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import enum
import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from fmmax import basis, utils, vector

# xx, xy, yx, yy, and zz components of permittivity or permeability.
_TensorComponents = Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]


@enum.unique
class Formulation(enum.Enum):
    """Enumerates supported Fourier modal method formulations."""

    FFT: str = "fft"
    JONES_DIRECT: str = vector.JONES_DIRECT
    JONES: str = vector.JONES
    NORMAL: str = vector.NORMAL
    POL: str = vector.POL


# -----------------------------------------------------------------------------
# Functions for computing Fourier convolution matrices for isotropic media.
# -----------------------------------------------------------------------------


def fourier_matrices_patterned_isotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Fourier convolution matrices for patterned nonmagnetic isotropic media.

    All matrices are forms of the Fourier convolution matrices defined in equation
    8 of [2012 Liu]. For vector formulations, the transverse permittivity matrix is
    of the form E2 given in equation 51 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, e.g. a vector formulation
            or the non-vector `FFT` formulation.

    Returns:
        inverse_z_permittivity_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix.
    """
    if formulation == Formulation.FFT:
        transverse_permittivity_matrix = _transverse_permittivity_fft(
            permittivity=permittivity,
            expansion=expansion,
        )
    else:
        transverse_permittivity_matrix = _transverse_permittivity_vector(
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=formulation,
        )

    _transform = functools.partial(
        fourier_convolution_matrix,
        expansion=expansion,
    )

    inverse_z_permittivity_matrix = _transform(1 / permittivity)
    z_permittivity_matrix = _transform(permittivity)
    return (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
    )


def _transverse_permittivity_fft(
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Computes the `eps` matrix from [2012 Liu] equation 15 using `fft` scheme.

    Args:
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.

    Returns:
        The transverse permittivity matrix.
    """
    eps_hat = fourier_convolution_matrix(permittivity, expansion)
    zeros = jnp.zeros_like(eps_hat)
    return jnp.block([[eps_hat, zeros], [zeros, eps_hat]])


def _transverse_permittivity_vector(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> jnp.ndarray:
    """Computes transverse permittivity matrix using a vector field methods.

    The transverse permittivity matrix is E2 given in equation 51 of [2012 Liu].

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
# Functions for computing Fourier convolution matrices for anisotropic media.
# -----------------------------------------------------------------------------


def fourier_matrices_patterned_anisotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: _TensorComponents,
    permeabilities: _TensorComponents,
    expansion: basis.Expansion,
    formulation: Formulation,
    vector_field_source: jnp.ndarray,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Return Fourier convolution matrices for patterned anisotropic media.

    The transverse permittivity and permeability matrices are of the form E1 given
    in equation 47 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: `(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)`, each having shape `(..., nx, ny)`.
        permeabilities: The elements of the permeability tensor: `(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)`, each having shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.
        vector_field_source: Array used to calculate the vector field, with shape
            matching the permittivities and permeabilities.

    Returns:
        inverse_z_permittivity_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix from
            equation 15 of [2012 Liu], computed in the manner prescribed by
            `fmm_formulation`.
        inverse_z_permeability_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permeability.
        z_permeability_matrix: The Fourier convolution matrix for the z-component
            of the permeability.
        transverse_permeability_matrix: The transverse permittivity matrix.
    """
    if formulation is Formulation.FFT:
        _matrix_fn = functools.partial(_fft_matrices_anisotropic, expansion=expansion)
    else:
        vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
        tx, ty = vector_fn(vector_field_source, primitive_lattice_vectors)
        _matrix_fn = functools.partial(
            _vector_matrices_anisotropic, tx=tx, ty=ty, expansion=expansion
        )

    return _matrix_fn(permittivities) + _matrix_fn(permeabilities)


def _fft_matrices_anisotropic(
    arrs: _TensorComponents,
    expansion: basis.Expansion,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes anisotropic layer Fourier convolution matrices for the `FFT` formulation."""
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)

    arr_xx, arr_xy, arr_yx, arr_yy, arr_zz = arrs

    inverse_z_matrix = _transform(1 / arr_zz)
    z_matrix = _transform(arr_zz)

    transverse_matrix = jnp.block(
        [
            [_transform(arr_xx), _transform(arr_xy)],
            [_transform(arr_yx), _transform(arr_yy)],
        ]
    )
    return inverse_z_matrix, z_matrix, transverse_matrix


def _vector_matrices_anisotropic(
    arrs: _TensorComponents,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis.Expansion,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes anisotropic layer Fourier convolution matrices for a vector formulation."""
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)

    arr_xx, arr_xy, arr_yx, arr_yy, arr_zz = arrs

    z_matrix = _transform(arr_zz)
    inverse_z_matrix = _transform(1 / arr_zz)

    # Obtain the tensorial quantity (permittivity or permeability) in the rotated
    # coordinate system.
    rotation_matrix = jnp.block([[ty, tx.conj()], [-tx, ty.conj()]])
    arr_tensor = jnp.block([[arr_yy, arr_yx], [arr_xy, arr_xx]])
    arr_rotated = jnp.linalg.solve(rotation_matrix, arr_tensor @ rotation_matrix)

    # Fourier transformed matrix. Note that the lower right matrix block uses the inverse
    # rule, whereas other blocks use the Laurent rule.
    fourier_arr_matrix = jnp.block(
        [
            [
                _transform(arr_rotated[..., 0, 0]),
                _transform(arr_rotated[..., 0, 1]),
            ],
            [
                _transform(arr_rotated[..., 1, 0]),
                jnp.linalg.inv(_transform(1 / arr_rotated[..., 1, 1])),
            ],
        ]
    )
    fourier_rotation_matrix = jnp.block(
        [
            [_transform(ty), _transform(tx.conj())],
            [_transform(-tx), _transform(ty.conj())],
        ]
    )

    # The transverse permittivity or permeability matrix of equation 47 in [2012 Liu].
    transverse_matrix = (
        fourier_rotation_matrix
        @ fourier_arr_matrix
        @ jnp.linalg.inv(fourier_rotation_matrix)
    )

    return inverse_z_matrix, z_matrix, transverse_matrix


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
