"""Functions that generate various matrices for the FMM problem.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Tuple

import jax.numpy as jnp

from fmmax import basis, fft, utils


def omega_script_k_matrix_patterned(
    wavelength: jnp.ndarray,
    z_permittivity_matrix: jnp.ndarray,
    transverse_permeability_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the omega-script-k matrix of equation 26 from [2012 Liu]."""
    # The script-k matrix from equation 19 of [2012 Liu].
    script_k_matrix = script_k_matrix_patterned(
        z_permittivity_matrix, transverse_wavevectors
    )
    angular_frequency = utils.angular_frequency_for_wavelength(wavelength)
    angular_frequency_squared = angular_frequency[..., jnp.newaxis, jnp.newaxis] ** 2
    return angular_frequency_squared * transverse_permeability_matrix - script_k_matrix


# -----------------------------------------------------------------------------
# Functions to compute the k- and script-k matrices.
# -----------------------------------------------------------------------------


def script_k_matrix_uniform(
    permittivity: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the uniform-layer script-k matrix from eq. 19 of [2012 Liu]."""
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    return jnp.block(
        [
            [
                utils.diag(ky / permittivity[..., jnp.newaxis] * ky),
                utils.diag(-ky / permittivity[..., jnp.newaxis] * kx),
            ],
            [
                utils.diag(-kx / permittivity[..., jnp.newaxis] * ky),
                utils.diag(kx / permittivity[..., jnp.newaxis] * kx),
            ],
        ]
    )


def script_k_matrix_patterned(
    z_permittivity_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the patterned-layer script-k matrix from eq. 19 of [2012 Liu]."""
    dtype = jnp.promote_types(z_permittivity_matrix, transverse_wavevectors)
    kx = transverse_wavevectors[..., 0].astype(dtype)
    ky = transverse_wavevectors[..., 1].astype(dtype)
    z_inv_kx = jnp.linalg.solve(z_permittivity_matrix.astype(dtype), utils.diag(kx))
    z_inv_ky = jnp.linalg.solve(z_permittivity_matrix.astype(dtype), utils.diag(ky))
    return jnp.block(
        [
            [ky[..., :, jnp.newaxis] * z_inv_ky, -ky[..., :, jnp.newaxis] * z_inv_kx],
            [-kx[..., :, jnp.newaxis] * z_inv_ky, kx[..., :, jnp.newaxis] * z_inv_kx],
        ]
    )


def k_matrix_patterned(
    z_permeability_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the k-matrix for patterned magnetic materials."""
    dtype = jnp.promote_types(z_permeability_matrix, transverse_wavevectors)
    kx = transverse_wavevectors[..., 0].astype(dtype)
    ky = transverse_wavevectors[..., 1].astype(dtype)
    z_inv_kx = jnp.linalg.solve(z_permeability_matrix.astype(dtype), utils.diag(kx))
    z_inv_ky = jnp.linalg.solve(z_permeability_matrix.astype(dtype), utils.diag(ky))
    return jnp.block(
        [
            [kx[..., :, jnp.newaxis] * z_inv_kx, kx[..., :, jnp.newaxis] * z_inv_ky],
            [ky[..., :, jnp.newaxis] * z_inv_kx, ky[..., :, jnp.newaxis] * z_inv_ky],
        ]
    )


# -----------------------------------------------------------------------------
# Functions for computing Fourier convolution matrices for isotropic media.
# -----------------------------------------------------------------------------


def transverse_permittivity_fft(
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
    eps_hat = fft.fourier_convolution_matrix(permittivity, expansion)
    zeros = jnp.zeros_like(eps_hat)
    return jnp.block([[eps_hat, zeros], [zeros, eps_hat]])


def transverse_permittivity_vector(
    permittivity: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Computes transverse permittivity matrix using a vector field methods.

    The transverse permittivity matrix is E2 given in equation 51 of [2012 Liu].

    Args:
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The `eps` matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    eps_hat = _transform(permittivity)
    zeros = jnp.zeros_like(eps_hat)
    eps_matrix = jnp.block([[eps_hat, zeros], [zeros, eps_hat]])

    eta = 1 / permittivity
    eta_hat = _transform(eta)
    delta_hat = eps_hat - jnp.linalg.inv(eta_hat)
    delta_matrix = jnp.block([[delta_hat, zeros], [zeros, delta_hat]])

    Pxx, Pxy, Pyx, Pyy = _tangent_terms(tx, ty)
    p_matrix = jnp.block(
        [[_transform(Pyy), _transform(Pyx)], [_transform(Pxy), _transform(Pxx)]]
    )
    return eps_matrix - delta_matrix @ p_matrix


def _tangent_terms(
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


def transverse_permittivity_fft_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permittivity matrix for anisotropic media using the fft scheme."""
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permittivity_yy), _transform(-permittivity_yx)],
            [_transform(-permittivity_xy), _transform(permittivity_xx)],
        ]
    )


def transverse_permeability_fft_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permeability matrix for anisotropic media using the fft scheme."""
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permeability_xx), _transform(permeability_xy)],
            [_transform(permeability_yx), _transform(permeability_yy)],
        ]
    )


def transverse_permittivity_vector_anisotropic(
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
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The transverse permittivity matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

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


def transverse_permeability_vector_anisotropic(
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
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The transverse permeability matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    # Define the real-space and Fourier transformed rotation matrices. The rotation
    # matrix is defined such that T [Ht, Hn]^T = [Hx, Hy], and differs from that
    # used for the transverse permittivity calculation.
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
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
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
