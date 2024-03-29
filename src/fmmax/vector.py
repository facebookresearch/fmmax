"""Functions related to tangent vector field generation.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp

from fmmax import basis, fft, utils

# Absolute tolerance for detecting whether a field is 1D. If the angle of the field at
# every point differs by less than this value from a reference value, the field is 1D.
_ATOL_1D_FIELD_ANGLE = 1e-2


def compute_field_jones_direct(
    arr: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute tangent vector field using the Jones direct method."""
    return compute_tangent_field(
        arr=arr,
        expansion=expansion,
        primitive_lattice_vectors=primitive_lattice_vectors,
        use_jones_direct=True,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
    )


def compute_field_pol(
    arr: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute tangent vector field using the Pol method."""
    return compute_tangent_field(
        arr=arr,
        expansion=expansion,
        primitive_lattice_vectors=primitive_lattice_vectors,
        use_jones_direct=False,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
    )


def compute_field_jones(
    arr: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute tangent vector field using the Jones method."""
    tx, ty = compute_tangent_field(
        arr=arr,
        expansion=expansion,
        primitive_lattice_vectors=primitive_lattice_vectors,
        use_jones_direct=False,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
    )
    jxjy = normalize_jones(jnp.stack([tx, ty], axis=-1))
    jx = jxjy[..., 0]
    jy = jxjy[..., 1]
    return jx, jy


def compute_field_normal(
    arr: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute tangent vector field using the Normal method."""
    tx, ty = compute_tangent_field(
        arr=arr,
        expansion=expansion,
        primitive_lattice_vectors=primitive_lattice_vectors,
        use_jones_direct=False,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
    )
    txty = normalize_elementwise(jnp.stack([tx, ty], axis=-1))
    tx = txty[..., 0]
    ty = txty[..., 1]
    return tx, ty


# -----------------------------------------------------------------------------
# Underlying functions used to compute all variants of the tangent fields.
# -----------------------------------------------------------------------------


def compute_tangent_field(
    arr: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    use_jones_direct: bool,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
    steps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the tangent vector field for `arr`.

    The calculation finds the minimum of a quadratic loss function using a single
    Newton iteration. Rather than optimizing the real-space tangent field, the
    Fourier coefficients are directly optimized.

    The tangent field has several properties or invariances:

      - The tangent field is independent of the scale of the unit cell; if the
        unit cell is uniformly scaled (e.g. by changing units from nm to microns),
        the vector field is unchanged.
      - The tangent field for a supercell (containing e.g. 2x2 unit cells) is
        identical to that of a single unit cell, so long as the number of terms in
        the Fourier expansion is increased correspondingly. Note that this means that
        the tangent field depends upon the number of terms in the Fourier expansion.
      - The tangent field is independent of the resolution of the discretized unit
        cell. That is, whether the permittivity distribution is specified with a
        `(100, 100)` or `(200, 200)` shaped array has no impact on the resulting field.
      - The tangent field is independent of the amplitude of the array from which it is
        obtained, e.g. the permittivity contrast.

    Args:
        arr: The array for which the normal vector field is sought.
        expansion: The Fourier expansion for which the field is to be optimized.
        primitive_lattice_vectors: Define the unit cell coordinates.
        use_jones_direct: Specifies whether the complex Jones field is to be sought.
        fourier_loss_weight: Determines the weight of the loss term penalizing
            Fourier terms corresponding to high frequencies. Should be positive.
        smoothness_loss_weight: Determines the weight of the loss term rewarding
            smoothness of the tangent field in real space. Should be positive.
        steps: The number of Newton iterations to carry out. Generally, the default
            single iteration is sufficient to obtain converged fields.

    Returns:
        The normal field, `(tx, ty)`.
    """
    arr = jax.lax.stop_gradient(arr)
    batch_shape = arr.shape[:-2]
    arr = utils.atleast_nd(arr, n=3)
    arr = arr.reshape((-1,) + arr.shape[-2:])

    primitive_lattice_vectors = basis.LatticeVectors(
        u=jnp.broadcast_to(
            primitive_lattice_vectors.u.reshape((-1, 2)),
            (arr.shape[0], 2),
        ),
        v=jnp.broadcast_to(
            primitive_lattice_vectors.v.reshape((-1, 2)),
            (arr.shape[0], 2),
        ),
    )

    field_fn = jax.vmap(
        functools.partial(
            _compute_tangent_field_no_batch,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
            steps=steps,
        ),
        in_axes=(0, None, 0),
    )
    field = field_fn(
        arr,
        expansion,
        primitive_lattice_vectors,
    )
    field = field.reshape(batch_shape + field.shape[-3:])
    tx = field[..., 0]
    ty = field[..., 1]
    return tx, ty


def _compute_tangent_field_no_batch(
    arr: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    use_jones_direct: bool,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
    steps: int,
) -> jnp.ndarray:
    """Compute the tangent vector field for `arr` with no batch dimensions."""
    assert primitive_lattice_vectors.u.shape == (2,)
    assert arr.ndim == 2
    grid_shape: Tuple[int, int] = arr.shape[-2:]  # type: ignore[assignment]

    # Rescale the weights so that a supercell containing multiple unit cells and
    # having appropriately more terms in the Fourier expansion yields a tangent
    # field identical to that obtained from just a single unit cell.
    fourier_loss_weight /= expansion.num_terms
    smoothness_loss_weight /= expansion.num_terms

    grad = compute_gradient(arr, primitive_lattice_vectors)
    gx = _filter_and_adjust_resolution(grad[..., 0], expansion)
    gy = _filter_and_adjust_resolution(grad[..., 1], expansion)
    grad = normalize(jnp.stack([gx, gy], axis=-1))

    elementwise_alignment_weight = _field_magnitude(grad)

    # Provide a dummy gradient for the 1D case, which avoids possible nans in the
    # Newton solve below. The tangent vector field will be manually specified.
    is_1d, grad_angle = _is_1d_field(grad)
    dummy_grad = jnp.broadcast_to(jnp.asarray([1, 1], dtype=complex), grad.shape)
    grad = jnp.where(is_1d, dummy_grad, grad)

    # Compute the target field with which the tangent field should be aligned.
    target_field = jnp.stack([grad[..., 1], -grad[..., 0]], axis=-1)
    target_field = normalize_elementwise(target_field)
    if use_jones_direct:
        target_field = normalize_jones(target_field)
        initial_field = target_field
    else:
        initial_field = normalize(jnp.stack([grad[..., 1], -grad[..., 0]], axis=-1))
    fourier_field = fft.fft(initial_field, expansion=expansion, axes=(-3, -2))

    flat_fourier_field = fourier_field.flatten()
    for _ in range(steps):
        _, jac, hessian = field_loss_value_jac_and_hessian(
            flat_fourier_field=flat_fourier_field,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
            target_field=target_field,
            elementwise_alignment_loss_weight=elementwise_alignment_weight,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )
        flat_fourier_field -= jnp.linalg.solve(hessian, jac.conj())
    fourier_field = flat_fourier_field.reshape((expansion.num_terms, 2))

    field = fft.ifft(fourier_field, expansion=expansion, shape=grid_shape, axis=-2)

    # Manually set the tangent field in the 1d case.
    field_1d = jnp.stack([jnp.sin(grad_angle), jnp.cos(grad_angle)])
    field = jnp.where(is_1d, field_1d, field)
    return normalize(field)


# -------------------------------------------------------------------------------------
# Normalization and transform functions.
# -------------------------------------------------------------------------------------


def normalize_jones(field: jnp.ndarray) -> jnp.ndarray:
    """Generates a Jones vector field following the "Jones" method of [2012 Liu]."""
    assert field.shape[-1] == 2
    field = normalize(field)
    magnitude = _field_magnitude(field)

    magnitude_near_zero = jnp.isclose(magnitude, 0.0)
    magnitude_safe = jnp.where(magnitude_near_zero, 1.0, magnitude)
    tx_norm = jnp.where(
        magnitude_near_zero,
        1 / jnp.sqrt(2),
        field[..., 0, jnp.newaxis] / magnitude_safe,
    )
    ty_norm = jnp.where(
        magnitude_near_zero,
        1 / jnp.sqrt(2),
        field[..., 1, jnp.newaxis] / magnitude_safe,
    )

    phi = jnp.pi / 8 * (1 + jnp.cos(jnp.pi * magnitude))
    theta = _angle(tx_norm + 1j * ty_norm)

    jx = jnp.exp(1j * theta) * (tx_norm * jnp.cos(phi) - ty_norm * 1j * jnp.sin(phi))
    jy = jnp.exp(1j * theta) * (ty_norm * jnp.cos(phi) + tx_norm * 1j * jnp.sin(phi))
    return jnp.concatenate([jx, jy], axis=-1)


def normalize_elementwise(field: jnp.ndarray) -> jnp.ndarray:
    """Normalize the elements of `field` to have magnitude `1` everywhere."""
    magnitude = _field_magnitude(field)
    magnitude_safe = jnp.where(jnp.isclose(magnitude, 0), 1, magnitude)
    return field / magnitude_safe


def normalize(field: jnp.ndarray) -> jnp.ndarray:
    """Normalize `field` so that it has maximum magnitude `1`."""
    max_magnitude = _max_field_magnitude(field)
    max_magnitude_safe = jnp.where(jnp.isclose(max_magnitude, 0), 1, max_magnitude)
    return field / max_magnitude_safe


def _field_magnitude(field: jnp.ndarray) -> jnp.ndarray:
    """Return the magnitude of `field`"""
    magnitude_squared = jnp.sum(jnp.abs(field) ** 2, axis=-1, keepdims=True)
    is_zero = magnitude_squared == 0
    magnitude_squared_safe = jnp.where(is_zero, 1.0, magnitude_squared)
    return jnp.where(is_zero, 0.0, jnp.sqrt(magnitude_squared_safe))


def _max_field_magnitude(field: jnp.ndarray) -> jnp.ndarray:
    """Returns the magnitude of the largest component in `field`."""
    return jnp.amax(_field_magnitude(field), axis=(-3, -2), keepdims=True)


def _angle(x: jnp.ndarray) -> jnp.ndarray:
    """Computes `angle(x)` with special logic for when `x` equals zero."""
    # Avoid taking the angle of an array with any near-zero elements, to avoid
    # `nan` in the gradients.
    is_near_zero = jnp.isclose(x, 0.0)
    x_safe = jnp.where(is_near_zero, (1.0 + 0.0j), x)
    return jnp.angle(x_safe)


def _filter_and_adjust_resolution(
    x: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Filter `x` and adjust its resolution for the given `expansion`."""
    y = fft.fft(x, expansion=expansion)
    min_shape = fft.min_array_shape_for_expansion(expansion)
    assert x.ndim == 2
    # Singleton dimensions remain singleton.
    doubled_min_shape = (
        min_shape[0] * (2 if x.shape[0] > 1 else 1),
        min_shape[1] * (2 if x.shape[1] > 1 else 1),
    )
    return fft.ifft(y, expansion=expansion, shape=doubled_min_shape)


def _is_1d_field(field: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Determine whether the field varies in one direction only."""
    ref_field = _field_at_max_magnitude(field)
    assert ref_field.shape == (2,)
    ref_angle = _angle(ref_field[0] + 1j * ref_field[1])
    angle = _angle(field[..., 0] + 1j * field[..., 1])
    magnitude = jnp.squeeze(_field_magnitude(field), axis=-1)
    is_1d = jnp.all(
        jnp.isclose(magnitude, 0.0)
        | jnp.isclose(angle, ref_angle - 2 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle - 1 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle + 1 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle + 2 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
    )
    # If one of the spatial dimensions is a singleton, the field is automatically 1D.
    assert field.ndim == 3
    is_1d = is_1d | (field.shape[0] == 1) | (field.shape[1] == 1)
    return is_1d, ref_angle


def _field_at_max_magnitude(field: jnp.ndarray) -> jnp.ndarray:
    """Return the field at the location where its magnitude is largest."""
    assert field.ndim == 3
    assert field.shape[-1] == 2
    magnitude = _field_magnitude(field)
    assert magnitude.ndim == 3
    assert magnitude.shape[-1] == 1
    magnitude = magnitude.flatten()
    idx = jnp.argmax(magnitude)
    field = field.reshape((-1, 2))
    return field[idx, :]


# -------------------------------------------------------------------------------------
# Loss function
# -------------------------------------------------------------------------------------


def field_loss_value_jac_and_hessian(
    flat_fourier_field: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    target_field: jnp.ndarray,
    elementwise_alignment_loss_weight: jnp.ndarray,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the value, Jacobian, and Hessian of the field loss."""
    assert flat_fourier_field.ndim == 1

    def fn(flat_fourier_field):
        fourier_field = jnp.reshape(flat_fourier_field, (-1, 2))
        value = _field_loss(
            fourier_field=fourier_field,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
            target_field=target_field,
            elementwise_alignment_loss_weight=elementwise_alignment_loss_weight,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )
        return value, value

    def jac_fn(flat_fourier_field):
        jac, value = jax.jacrev(fn, has_aux=True)(flat_fourier_field)
        return jac, (value, jac)

    hessian, (value, jac) = jax.jacrev(jac_fn, has_aux=True, holomorphic=True)(
        flat_fourier_field
    )

    return value, jac, hessian


def _field_loss(
    fourier_field: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    target_field: jnp.ndarray,
    elementwise_alignment_loss_weight: jnp.ndarray,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> jnp.ndarray:
    """Compute loss that favors smooth"""
    shape: Tuple[int, int] = target_field.shape[-3:-1]  # type: ignore[assignment]
    field = fft.ifft(
        y=fourier_field,
        expansion=expansion,
        shape=shape,
        axis=-2,
    )
    loss = _alignment_loss(field, target_field, elementwise_alignment_loss_weight)

    # Avoid calculating the fourier loss and smoothness loss if their weights are zero.
    # On some platforms, including a smoothness loss can signifcantly slow the compile
    # times, and so this optimization increases performance.
    with jax.ensure_compile_time_eval():
        assert jnp.size(fourier_loss_weight) == 1
        assert jnp.size(smoothness_loss_weight) == 1
        use_fourier_loss = fourier_loss_weight > 0
        use_smoothness_loss = smoothness_loss_weight > 0

    if use_fourier_loss:
        loss += fourier_loss_weight * _fourier_loss(
            fourier_field, expansion, primitive_lattice_vectors
        )

    if use_smoothness_loss:
        loss += smoothness_loss_weight * _smoothness_loss(
            field, primitive_lattice_vectors
        )

    return loss


def _alignment_loss(
    field: jnp.ndarray,
    target_field: jnp.ndarray,
    elementwise_alignment_loss_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Compute loss that penalizes differences between `field` and `target_field`."""
    assert elementwise_alignment_loss_weight.ndim == field.ndim
    elementwise_loss = jnp.sum(
        jnp.abs(field - target_field) ** 2, axis=-1, keepdims=True
    )
    return jnp.mean(elementwise_alignment_loss_weight * elementwise_loss)


def _fourier_loss(
    fourier_field: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
) -> jnp.ndarray:
    """Compute loss that penalizes high frequency Fourier components.

    The loss is scaled for the size of the unit cell, i.e. two unit cells with
    identical `fourier_field` and scaled `primitive_lattice_vectors` will have
    identical loss.

    Args:
        fourier_field: The Fourier field for which the loss is sought.
        expansion: The Fourier expansion for the field.
        primitive_lattice_vectors: Defines the unit cell coordinates.

    Returns:
        The scalar fourier loss value.
    """
    transverse_wavevectors = basis.transverse_wavevectors(
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        in_plane_wavevector=jnp.zeros((2,)),
    )

    basis_vectors = jnp.stack(
        [primitive_lattice_vectors.u, primitive_lattice_vectors.v],
        axis=-1,
    )
    area = jnp.abs(jnp.linalg.det(basis_vectors))

    kt = jnp.linalg.norm(transverse_wavevectors, axis=-1) * jnp.sqrt(area)
    return jnp.sum(jnp.abs(fourier_field) ** 2 * kt[..., jnp.newaxis] ** 2)


def _smoothness_loss(
    field: jnp.ndarray, basis_vectors: basis.LatticeVectors
) -> jnp.ndarray:
    """Compute loss associated with smoothness of `field`."""
    grads = _vector_field_forward_difference_gradient(field, basis_vectors)
    return jnp.mean(jnp.abs(jnp.asarray(grads)) ** 2)


# -------------------------------------------------------------------------------------
# Gradient calculation and transformation.
# -------------------------------------------------------------------------------------


def compute_gradient(
    arr: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
) -> jnp.ndarray:
    """Computes the gradient of `arr`.

    The gradient is scaled for the size of the unit cell, i.e. two unit cells with
    identical `arr` and scaled `primitive_lattice_vectors` will have identical
    gradient.

    Args:
        arr: The array for which the gradient is sought.
        primitive_lattice_vectors: Defines the unit cell coordinates.

    Returns:
        The gradient, with shape `arr.shape + (2,)`.
    """
    basis_vectors = jnp.stack(
        [primitive_lattice_vectors.u, primitive_lattice_vectors.v],
        axis=-1,
    )

    area = jnp.abs(jnp.linalg.det(basis_vectors))
    basis_vectors /= jnp.sqrt(area)

    batch_dims = arr.ndim - 2
    pad_width = tuple([(0, 0)] * batch_dims + [(1, 1)] * 2)
    arr_padded = jnp.pad(arr, pad_width, mode="wrap")

    axes = tuple(range(-2, 0))
    partial_grad: List[jnp.ndarray]
    partial_grad = jnp.gradient(arr_padded, axis=axes)  # type: ignore[assignment]
    partial_grad = [_unpad(g, pad_width) for g in partial_grad]
    partial_grad = [g * arr.shape[ax] for g, ax in zip(partial_grad, axes)]
    return _transform_gradient(jnp.stack(partial_grad, axis=-1), basis_vectors)


def _vector_field_forward_difference_gradient(
    field: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the gradient of a vector field by forward difference.

    The returned gradients are,

        grad = (grad_field_x, grad_field_y)

    where

        grad_field_x = stack([dfield_x / dx, dfield_y / dy], axis=-1)

    Args:
        field: The field for which the gradient is sought.
        primitive_lattice_vectors: Defines the unit cell coordinates.

    Returns:
        Tuple containing the gradients, each with the same shape as `field`.
    """
    basis_vectors = jnp.stack(
        [primitive_lattice_vectors.u, primitive_lattice_vectors.v],
        axis=-1,
    )

    area = jnp.abs(jnp.linalg.det(basis_vectors))
    basis_vectors /= jnp.sqrt(area)
    return (
        _scalar_field_forward_difference_gradient(field[..., 0], basis_vectors),
        _scalar_field_forward_difference_gradient(field[..., 1], basis_vectors),
    )


def _scalar_field_forward_difference_gradient(
    field: jnp.ndarray,
    basis_vectors: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the gradient of a scalar field by forward difference.

    Args:
        field: The scalar field for which the forward-difference gradient is sought.
        basis_vectors: The vectors defining the space in which `field` is defined.

    Returns:
        The gradient, with shape `field.shape + (dimensions,)`.
    """
    dimension = basis_vectors.shape[-1]
    assert basis_vectors.shape[-2:] == (dimension, dimension)
    axes = range(field.ndim - dimension, field.ndim)
    diffs = [_periodic_forward_difference(field, axis=ax) for ax in axes]
    partial_grad = jnp.stack(diffs, axis=-1)
    return _transform_gradient(partial_grad, basis_vectors)


def _periodic_forward_difference(field: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Computes the forward difference for `field` along `axis`."""
    diff = jnp.roll(field, shift=-1, axis=axis) - field
    return diff * field.shape[axis]


def _transform_gradient(
    partial_grad: jnp.ndarray,
    basis_vectors: jnp.ndarray,
) -> jnp.ndarray:
    """Compute gradient from partial gradient with arbitrary basis vectors."""
    # https://en.wikipedia.org/wiki/Gradient#General_coordinates
    metric_tensor = _metric_tensor(basis_vectors).astype(partial_grad.dtype)
    inverse_metric_tensor = jnp.linalg.inv(metric_tensor)
    result: jnp.ndarray = partial_grad @ inverse_metric_tensor @ basis_vectors.T
    return result


def _metric_tensor(basis_vectors: jnp.ndarray) -> jnp.ndarray:
    """Compute the metric tensor for a non-Cartesian, non-orthogonal basis."""
    return basis_vectors.T @ basis_vectors


def _unpad(arr: jnp.ndarray, pad_width: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """Undoes a pad operation."""
    slices = [slice(lo, size - hi) for (lo, hi), size in zip(pad_width, arr.shape)]
    return arr[tuple(slices)]


# -----------------------------------------------------------------------------
# Library of available vector field generating schemes.
# -----------------------------------------------------------------------------


JONES_DIRECT: str = "jones_direct"
JONES: str = "jones"
NORMAL: str = "normal"
POL: str = "pol"

JONES_DIRECT_FOURIER: str = "jones_direct_fourier"
JONES_FOURIER: str = "jones_fourier"
NORMAL_FOURIER: str = "normal_fourier"
POL_FOURIER: str = "pol_fourier"

FOURIER_LOSS_WEIGHT: float = 0.2
SMOOTHNESS_LOSS_WEIGHT: float = 2.0

FOURIER_LOSS_WEIGHT_JONES_DIRECT: float = 0.05
SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT: float = 0.5

VectorFn = Callable[
    [jnp.ndarray, basis.Expansion, basis.LatticeVectors],
    Tuple[jnp.ndarray, jnp.ndarray],
]

VECTOR_FIELD_SCHEMES: Dict[str, VectorFn] = {
    JONES_DIRECT: functools.partial(
        compute_field_jones_direct,
        fourier_loss_weight=0.0,
        smoothness_loss_weight=SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT,
    ),
    JONES: functools.partial(
        compute_field_jones,
        fourier_loss_weight=0.0,
        smoothness_loss_weight=SMOOTHNESS_LOSS_WEIGHT,
    ),
    NORMAL: functools.partial(
        compute_field_normal,
        fourier_loss_weight=0.0,
        smoothness_loss_weight=SMOOTHNESS_LOSS_WEIGHT,
    ),
    POL: functools.partial(
        compute_field_pol,
        fourier_loss_weight=0.0,
        smoothness_loss_weight=SMOOTHNESS_LOSS_WEIGHT,
    ),
    JONES_DIRECT_FOURIER: functools.partial(
        compute_field_jones_direct,
        fourier_loss_weight=FOURIER_LOSS_WEIGHT_JONES_DIRECT,
        smoothness_loss_weight=0.0,
    ),
    JONES_FOURIER: functools.partial(
        compute_field_jones,
        fourier_loss_weight=FOURIER_LOSS_WEIGHT,
        smoothness_loss_weight=0.0,
    ),
    NORMAL_FOURIER: functools.partial(
        compute_field_normal,
        fourier_loss_weight=FOURIER_LOSS_WEIGHT,
        smoothness_loss_weight=0.0,
    ),
    POL_FOURIER: functools.partial(
        compute_field_pol,
        fourier_loss_weight=FOURIER_LOSS_WEIGHT,
        smoothness_loss_weight=0.0,
    ),
}
