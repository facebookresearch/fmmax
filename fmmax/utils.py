"""Defines several utility functions.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax.experimental import host_callback

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


def magnitude(tx: jnp.ndarray, ty: jnp.ndarray) -> jnp.ndarray:
    """Computes elementwise magnitude of the vector field defined by `(tx, ty)`.

    This method computes the magnitude with special logic that avoides `nan`
    gradients when the magnitude is zero.

    Args:
        tx: Array giving the x-component of the vector field.
        ty: Array giving the y-component of the vector field.

    Returns:
        Array giving the vector magnitude.
    """
    # Avoid taking the square root of an array with any zero elements, to avoid
    # `nan` in the gradients.
    magnitude_squared = jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
    is_zero = magnitude_squared == 0
    magnitude_squared_safe = jnp.where(is_zero, 1.0, magnitude_squared)
    return jnp.where(is_zero, 0.0, jnp.sqrt(magnitude_squared_safe))


def angle(x: jnp.ndarray) -> jnp.ndarray:
    """Computes `angle(x)` with special logic for when `x` equals zero."""
    # Avoid taking the angle of an array with any near-zero elements, to avoid
    # `nan` in the gradients.
    is_near_zero = jnp.isclose(x, 0.0)
    x_safe = jnp.where(is_near_zero, (1.0 + 0.0j), x)
    return jnp.angle(x_safe)


def resample(
    x: jnp.ndarray,
    shape: Tuple[int, ...],
    method=jax.image.ResizeMethod,
) -> jnp.ndarray:
    """Resamples `x` to have the specified `shape`.

    The algorithm first upsamples `x` so that the pixels in the output image are
    comprised of an integer number of pixels in the upsampled `x`, and then
    performs box downsampling.

    Args:
        x: The array to be resampled.
        shape: The shape of the output array.
        method: The method used to resize `x` prior to box downsampling.

    Returns:
        The resampled array.
    """
    if x.ndim != len(shape):
        raise ValueError(
            f"`shape` must have length matching number of dimensions in `x`, "
            f"but got {shape} when `x` had shape {x.shape}."
        )

    factor = [int(onp.ceil(dx / d)) for dx, d in zip(x.shape, shape)]
    upsampled_shape = tuple([d * f for d, f in zip(shape, factor)])

    x_upsampled = jax.image.resize(
        image=x,
        shape=upsampled_shape,
        method=method,
    )

    return box_downsample(x_upsampled, shape)


def box_downsample(x: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Downsamples `x` to a coarser resolution array using box downsampling.

    Box downsampling forms nonoverlapping windows and simply averages the
    pixels within each window. For example, downsampling `(0, 1, 2, 3, 4, 5)`
    with a factor of `2` yields `(0.5, 2.5, 4.5)`.

    Args:
        x: The array to be downsampled.
        shape: The shape of the output array; each axis dimension must evenly
            divide the corresponding axis dimension in `x`.

    Returns:
        The output array with shape `shape`.
    """
    if x.ndim != len(shape) or any([(d % s) != 0 for d, s in zip(x.shape, shape)]):
        raise ValueError(
            f"Each axis of `shape` must evenly divide the corresponding axis "
            f"dimension in `x`, but got shape {shape} when `x` has shape "
            f"{x.shape}."
        )
    shape = sum([(s, d // s) for d, s in zip(x.shape, shape)], ())
    axes = list(range(1, 2 * x.ndim, 2))
    x = x.reshape(shape)
    return jnp.mean(x, axis=axes)


# -----------------------------------------------------------------------------
# Functions related to convolutions and kernels.
# -----------------------------------------------------------------------------


def padded_conv(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    padding_mode: str,
) -> jnp.ndarray:
    """Convolves `x` with `kernel`, using padding with the specified mode.

    Before the convolution, `x` is padded using the specified padding mode.

    Args:
        x: The source array.
        kernel: The rank-2 convolutional kernel.
        padding_mode: One of the padding modes supported by `jnp.pad`.

    Returns:
        The result of the convolution.
    """
    assert kernel.ndim == 2
    pad_ij = [(s // 2 - (s + 1) % 2, s // 2) for s in kernel.shape]
    batch_dims = x.ndim - 2
    batch_shape = x.shape[:-2]
    padding = [(0, 0)] * batch_dims + pad_ij
    x = jnp.pad(x, padding, mode=padding_mode)
    # Flatten the batch dimensions and add a dummy channel dimension.
    x = x.reshape((-1,) + x.shape[-2:])
    x = jnp.expand_dims(x, -3)
    y = jax.lax.conv_general_dilated(
        x,
        kernel[jnp.newaxis, jnp.newaxis, :, :].astype(x.dtype),
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    # Remove the dummy channel dimension, and restore the original batch shape.
    y = jnp.squeeze(y, axis=-3)
    y = y.reshape(batch_shape + y.shape[-2:])
    return y


def gaussian_kernel(shape: Tuple[int, int], fwhm: float) -> jnp.ndarray:
    """Generates a Gaussian kernel with the specified shape."""
    x = jnp.arange(shape[0])[:, jnp.newaxis]
    y = jnp.arange(shape[1])[jnp.newaxis, :]
    # Ensure that the center is on a pixel, not between pixels.
    x_center = (shape[0] - 0.5) // 2
    y_center = (shape[1] - 0.5) // 2
    distance_squared = (x - x_center) ** 2 + (y - y_center) ** 2
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))

    # When `fwhm` is zero, we want to return a one-hot array.
    fwhm_is_zero = fwhm == 0.0
    denom_safe = jnp.where(fwhm_is_zero, 1.0, 2 * sigma**2)
    return jnp.where(
        fwhm_is_zero,
        (distance_squared == 0).astype(float),
        jnp.exp(-distance_squared / denom_safe),
    )


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
    eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, complex)

    def _eig_cpu(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        with jax.default_device(jax.devices("cpu")[0]):
            return jax.jit(jnp.linalg.eig)(matrix)

    return host_callback.call(
        _eig_cpu,
        matrix.astype(complex),
        result_shape=(eigenvalues_shape, eigenvectors_shape),
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

    # Compute the F-matrix, from equation 5 of [2021 Colburn]. This applies a
    # Lorentzian broadening to the matrix `f = 1 / (eigenvalues[i] - eigenvalues[j])`.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    f_broadened = (eigenvalues_i - eigenvalues_j) / (
        (eigenvalues_i - eigenvalues_j) ** 2 + eps
    )

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
