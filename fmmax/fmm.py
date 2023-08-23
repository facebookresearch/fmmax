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

    The transverse permittivity matrix E is defined as,

        [-Dy, Dx]^T = E [-Ey, Ex]^T

    while the transverse permeability matrix M is defined as,

        [Bx, By]^T = M [Hx, Hy]^T

    The Fourier factorization is done as for E1 given in equation 47 of [2012 Liu].

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
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)

    if formulation is Formulation.FFT:
        _transverse_permittivity_fn = _transverse_permittivity_fft_anisotropic
        _transverse_permeability_fn = _transverse_permeability_fft_anisotropic
    else:
        vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
        tx, ty = vector_fn(vector_field_source, primitive_lattice_vectors)
        _transverse_permittivity_fn = functools.partial(
            _transverse_permittivity_vector_anisotropic, tx=tx, ty=ty
        )
        _transverse_permeability_fn = functools.partial(
            _transverse_permeability_vector_anisotropic, tx=tx, ty=ty
        )

    (
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
    ) = permittivities
    inverse_z_permittivity_matrix = _transform(1 / permittivity_zz)
    z_permittivity_matrix = _transform(permittivity_zz)
    transverse_permittivity_matrix = _transverse_permittivity_fn(
        permittivity_xx=permittivity_xx,
        permittivity_xy=permittivity_xy,
        permittivity_yx=permittivity_yx,
        permittivity_yy=permittivity_yy,
        expansion=expansion,
    )

    (
        permeability_xx,
        permeability_xy,
        permeability_yx,
        permeability_yy,
        permeability_zz,
    ) = permeabilities
    inverse_z_permeability_matrix = _transform(1 / permeability_zz)
    z_permeability_matrix = _transform(permeability_zz)
    transverse_permeability_matrix = _transverse_permeability_fn(
        permeability_xx=permeability_xx,
        permeability_xy=permeability_xy,
        permeability_yx=permeability_yx,
        permeability_yy=permeability_yy,
        expansion=expansion,
    )
    return (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        inverse_z_permeability_matrix,
        z_permeability_matrix,
        transverse_permeability_matrix,
    )


def _transverse_permittivity_fft_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permittivity matrix for anisotropic media using the fft scheme."""
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permittivity_yy), _transform(-permittivity_yx)],
            [_transform(-permittivity_xy), _transform(permittivity_xx)],
        ]
    )


def _transverse_permeability_fft_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permeability matrix for anisotropic media using the fft scheme."""
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permeability_xx), _transform(permeability_xy)],
            [_transform(permeability_yx), _transform(permeability_yy)],
        ]
    )


def _transverse_permittivity_vector_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permittivity matrix with a vector scheme.

    The transverse permittivity matrix E relates the electric and electric displacement
    fields, such that

        [-Dy, Dx]^T = E [-Ey, Ex]^T

    Args:
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape `(..., nx, ny)`.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        permittivity_zz: The zz-component of the permittivity tensor.
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The transverse permittivity matrix.
    """
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)

    # Define the real-space and Fourier transformed rotation matrices. The rotation
    # matrix is defined such that T [Et, En]^T = [-Ey, Ex].
    (
        rotation_matrix,
        fourier_rotation_matrix,
        fourier_inverse_rotation_matrix,
    ) = _rotation_matrices(
        t00=ty,
        t01=tx.conj(),
        t10=-tx,
        t11=ty.conj(),
        expansion=expansion,
    )

    # Obtain the permittivity tensor for rotated coordinates, i.e. coordinates where
    # unit vectors are tangent and normal to the vector field defined by `(tx, ty)`.
    permittivity_tensor = jnp.block(
        [
            [
                permittivity_yy[..., jnp.newaxis, jnp.newaxis],
                -permittivity_yx[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                -permittivity_xy[..., jnp.newaxis, jnp.newaxis],
                permittivity_xx[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )
    assert rotation_matrix.shape[-2:] == permittivity_tensor.shape[-2:] == (2, 2)
    rotated_permittivity_tensor = jnp.linalg.solve(
        rotation_matrix, permittivity_tensor @ rotation_matrix
    )

    # The Fourier permittivity matrix is the central matrix in equation 45 of [2012 Liu].
    # The top left block relates the tangential E and D fields, while the bottom right
    # block relates the normal E and D fields. Consequently, these use Laurent's rule and
    # the inverse rule, respectively. Note that the non-diagonal blocks also use Laurent's
    # rule, as these tensor elements may be zero, making the inverse rule problematic.
    fourier_permittivity_matrix = jnp.block(
        [
            [
                _transform(rotated_permittivity_tensor[..., 0, 0]),
                _transform(rotated_permittivity_tensor[..., 0, 1]),
            ],
            [
                _transform(rotated_permittivity_tensor[..., 1, 0]),
                jnp.linalg.inv(_transform(1 / rotated_permittivity_tensor[..., 1, 1])),
            ],
        ]
    )
    return (
        fourier_rotation_matrix
        @ fourier_permittivity_matrix
        @ fourier_inverse_rotation_matrix
    )


def _transverse_permeability_vector_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permeability matrix with a vector scheme.

    The transverse permeability matrix M relates the magnetic and magnetic flux
    density fields, such that

        [Bx, Bx]^T = M [Hx, Hy]^T

    Important differences from the `_transverse_permittivity_vector_anisotropic`
    function result from the different definitions of E and M matrices.

    Args:
        permeability_xx: The xx-component of the permeability tensor, with
            shape `(..., nx, ny)`.
        permeability_xy: The xy-component of the permeability tensor.
        permeability_yx: The yx-component of the permeability tensor.
        permeability_yy: The yy-component of the permeability tensor.
        permeability_zz: The zz-component of the permeability tensor.
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The transverse permeability matrix.
    """
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)

    # Define the real-space and Fourier transformed rotation matrices. The rotation
    # matrix is defined such that T [Ht, Hn]^T = [Hx, Hy], and differs from that
    # used for the transverse permittiity calculation.
    (
        rotation_matrix,
        fourier_rotation_matrix,
        fourier_inverse_rotation_matrix,
    ) = _rotation_matrices(
        t00=tx,
        t01=-ty.conj(),
        t10=ty,
        t11=tx.conj(),
        expansion=expansion,
    )

    # Obtain the permeability tensor for rotated coordinates, i.e. coordinates where
    # unit vectors are tangent and normal to the vector field defined by `(tx, ty)`.
    permeability_tensor = jnp.block(
        [
            [
                permeability_xx[..., jnp.newaxis, jnp.newaxis],
                permeability_xy[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                permeability_yx[..., jnp.newaxis, jnp.newaxis],
                permeability_yy[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )
    assert rotation_matrix.shape[-2:] == permeability_tensor.shape[-2:] == (2, 2)
    rotated_permeability_tensor = jnp.linalg.solve(
        rotation_matrix, permeability_tensor @ rotation_matrix
    )

    # The Fourier permeability matrix is analogous to the central matrix in equation 45
    # of [2012 Liu], in that its top-left block relates the tangential magnetic and
    # magnetic flux density fields, and the bottom-right block relates the normal
    # fields. Consequently, these use Laurent's rule and the inverse rule, respectively.
    # Note that the non-diagonal blocks also use Laurent's rule, as these tensor elements
    # may be zero, making the inverse rule problematic.
    fourier_permeability_matrix = jnp.block(
        [
            [
                _transform(rotated_permeability_tensor[..., 0, 0]),  # tt
                _transform(rotated_permeability_tensor[..., 0, 1]),  # tn
            ],
            [
                _transform(rotated_permeability_tensor[..., 1, 0]),  # nt
                jnp.linalg.inv(
                    _transform(1 / rotated_permeability_tensor[..., 1, 1])
                ),  # nn
            ],
        ]
    )
    return (
        fourier_rotation_matrix
        @ fourier_permeability_matrix
        @ fourier_inverse_rotation_matrix
    )


def _rotation_matrices(
    t00: jnp.ndarray,
    t01: jnp.ndarray,
    t10: jnp.ndarray,
    t11: jnp.ndarray,
    expansion: basis.Expansion,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate real-space and Fourier transformed rotation matrices."""
    rotation_matrix = jnp.block(
        [
            [
                t00[..., jnp.newaxis, jnp.newaxis],
                t01[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                t10[..., jnp.newaxis, jnp.newaxis],
                t11[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )

    # Blockwise Fourier transform of the rotation matrix.
    _transform = functools.partial(fourier_convolution_matrix, expansion=expansion)
    fourier_rotation_matrix = jnp.block(
        [
            [_transform(t00), _transform(t01)],
            [_transform(t10), _transform(t11)],
        ]
    )

    # Blockwise Fourier transform of the inverse rotation matrix.
    denom = jnp.abs(t00 * t11 - t10 * t01)
    denom_safe = jnp.where(jnp.isclose(denom, 0), 1.0, denom)
    fourier_inverse_rotation_matrix = jnp.block(
        [
            [_transform(t11 / denom_safe), _transform(-t01 / denom_safe)],
            [_transform(-t10 / denom_safe), _transform(t00 / denom_safe)],
        ]
    )
    return rotation_matrix, fourier_rotation_matrix, fourier_inverse_rotation_matrix


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
