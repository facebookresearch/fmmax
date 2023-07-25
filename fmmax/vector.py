"""Functions related to tangent vector field generation.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Any, Callable, Dict, Tuple

import jax
import jax.example_libraries.optimizers as jopt
import jax.numpy as jnp
import numpy as onp

from fmmax import basis, utils

PyTree = Any


def normalized_vector_field(
    arr: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    vector_fn: Callable,
    normalize_fn: Callable,
    resize_max_dim: int,
    resize_method: jax.image.ResizeMethod,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates a normalized tangent vector field according to the specified method.

    Some `vector_fn` can be computationally expensive, and so this function
    resizes the input `arr` so that the maximum size of the two trailing
    dimensions is `resize_max_dim`. When `arr` is smaller than the maximum
    size, no resampling is performed.

    The tangent fields are then computed for this resized array, and then
    resized again to obtain fields at the original resolution.

    Args:
        arr: The array for which the tangent vector field is sought.
        primitive_lattice_vectors: Define the unit cell coordinates.
        vector_fn: Function used to generate the vector field.
        normalize_fn: Function used to normalize the vector field.
        resize_max_dim: Determines the size of the array for which the tangent
            vector field is computed; `arr` is resized so that it has a maximum
            size of `vector_arr_size` along any dimension.
        resize_method: Method used in scaling `arr` prior to calculating the
            tangent vector field.

    Returns:
        The normalized vector field.
    """

    shape = arr.shape
    resize_factor = resize_max_dim / max(shape[-2:])

    if resize_factor < 1:
        assert any(d > resize_max_dim for d in shape[-2:])
        coarse_shape = shape[:-2] + tuple(
            [int(onp.ceil(d * resize_factor)) for d in shape[-2:]]
        )
        assert len(coarse_shape) == arr.ndim
        arr = utils.resample(arr, shape=coarse_shape, method=resize_method)

    tu, tv = vector_fn(arr)
    assert tu.shape == tv.shape == arr.shape

    if resize_factor < 1:
        tu = jax.image.resize(tu, shape, method=resize_method)
        tv = jax.image.resize(tv, shape, method=resize_method)
        assert tu.shape == tv.shape == shape

    tx, ty = change_vector_field_basis(
        tu,
        tv,
        u=primitive_lattice_vectors.u,
        v=primitive_lattice_vectors.v,
        x=basis.X,
        y=basis.Y,
    )
    return normalize_fn(tx, ty)


def change_vector_field_basis(
    tu: jnp.ndarray,
    tv: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Changes the basis for a vector field.

    Specifically, given a field with amplitudes `(tu, tv)` of the basis
    vectors `(u, v)`, this function computes the amplitudes for the basis
    vectors `(x, y)`.

    Args:
        tu: The amplitude of the first basis vector in the original basis.
        tv: The amplitude of the second basis vector in the original basis.
        u: The first vector of the original basis.
        v: The second vector of the original basis.
        x: The first vector of the new basis.
        y: The second vector of the new basis.

    Returns:
        The field `(tx, ty)` in the new basis.
    """
    xy = jnp.stack([x, y], axis=-1)
    uxy = jnp.linalg.solve(xy, u)
    vxy = jnp.linalg.solve(xy, v)
    tx = tu * uxy[..., 0] + tv * vxy[..., 0]
    ty = tu * uxy[..., 1] + tv * vxy[..., 1]
    return tx, ty


def compute_gradient(arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the gradient of `arr` with respect to `x` and `y`.

    This function uses periodic boundary conditions. The `x` and `y`
    dimensions correspond to the trailing axes of `arr`.

    Args:
        arr: The array whose gradient is sought.

    Returns:
        The `(gx, gy)` gradients along the x- and y- directions.
    """
    batch_dims = arr.ndim - 2
    padding = tuple([(0, 0)] * batch_dims + [(1, 1), (1, 1)])
    arr_padded = jnp.pad(arr, padding, mode="wrap")
    gradient_x = jnp.gradient(arr_padded, axis=-2)[..., 1:-1, 1:-1]
    gradient_y = jnp.gradient(arr_padded, axis=-1)[..., 1:-1, 1:-1]
    return gradient_x, gradient_y


# -----------------------------------------------------------------------------
# Functions related to vector field normalization.
# -----------------------------------------------------------------------------


def normalize_normal(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalizes the tangent vector field using the "Normal" method of [2012 Liu]."""
    magnitude = utils.magnitude(tx, ty)
    magnitude_safe = jnp.where(jnp.isclose(magnitude, 0), 1, magnitude)
    tx /= magnitude_safe
    ty /= magnitude_safe
    return tx, ty


def normalize_pol(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalizes the tangent vector field using the "Pol" method of [2012 Liu]."""
    magnitude = utils.magnitude(tx, ty)
    max_magnitude = jnp.amax(magnitude, axis=(-2, -1), keepdims=True)
    max_magnitude_safe = jnp.where(jnp.isclose(max_magnitude, 0.0), 1.0, max_magnitude)
    tx /= max_magnitude_safe
    ty /= max_magnitude_safe
    return tx, ty


def normalize_jones(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates a Jones vector field following the "Jones" method of [2012 Liu]."""
    magnitude = utils.magnitude(tx, ty)
    max_magnitude = jnp.amax(magnitude, axis=(-2, -1), keepdims=True)
    max_magnitude_safe = jnp.where(jnp.isclose(max_magnitude, 0.0), 1.0, max_magnitude)
    tx /= max_magnitude_safe
    ty /= max_magnitude_safe
    magnitude = magnitude / max_magnitude_safe

    magnitude_near_zero = jnp.isclose(magnitude, 0.0)
    magnitude_safe = jnp.where(magnitude_near_zero, 1.0, magnitude)
    tx_norm = jnp.where(magnitude_near_zero, 1 / jnp.sqrt(2), tx / magnitude_safe)
    ty_norm = jnp.where(magnitude_near_zero, 1 / jnp.sqrt(2), ty / magnitude_safe)

    phi = jnp.pi / 8 * (1 + jnp.cos(jnp.pi * magnitude))
    theta = utils.angle(tx_norm + 1j * ty_norm)

    jx = jnp.exp(1j * theta) * (tx_norm * jnp.cos(phi) - ty_norm * 1j * jnp.sin(phi))
    jy = jnp.exp(1j * theta) * (ty_norm * jnp.cos(phi) + tx_norm * 1j * jnp.sin(phi))

    return jx, jy


# -----------------------------------------------------------------------------
# Functions related to vector field generation by minimizing a functional.
# -----------------------------------------------------------------------------


def tangent_field(
    arr: jnp.ndarray,
    use_jones: bool,
    optimizer: jopt.Optimizer,
    alignment_weight: float,
    smoothness_weight: float,
    steps_dim_multiple: int,
    smoothing_kernel: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes a real or complex tangent vector field.

    The field is tangent to the interfaces of features in `arr`, and varies smoothly
    between interfaces. It is obtained by optimization which minimizes a functional
    of the field which favors alignment with interfaces of features in `arr` as well
    as smoothness of the field. The maximum magnitude of the computed field is `1`.

    The tangent field is complex when `use_jones` is `True`, and real otherwise. Real
    fields are suitable for normalization using methods in this module. The complex
    field obtained when `use_jones` is `True` requires no normalization.

    Args:
        arr: The array for which the tangent field is sought.
        use_jones: Specifies whether a complex Jones field or a real tangent vector
            field is sought.
        optimizer: The optimizer used to minimize the functional.
        alignment_weight: The weight of an alignment term in the functional. Larger
            values will reward alignment with interfaces of features in `arr`.
        smoothness_weight: The weight of a smoothness term in the functional. Larger
            values will reward a smoother tangent field.
        steps_dim_multiple: Controls the number of steps in the optimization. The
            number of steps is the product of `steps_dim_multiple` and the dimension
            of the largest of the two trailing axes of `arr`.
        smoothing_kernel: Kernel used to smooth `arr` prior to the computation.

    Returns:
        The tangent vector fields, `(tx, ty)`.
    """
    tx, ty, _ = _tangent_field_with_loss(
        arr=arr,
        use_jones=use_jones,
        optimizer=optimizer,
        alignment_weight=alignment_weight,
        smoothness_weight=smoothness_weight,
        steps_dim_multiple=steps_dim_multiple,
        smoothing_kernel=smoothing_kernel,
    )
    return tx, ty


def _tangent_field_with_loss(
    arr: jnp.ndarray,
    use_jones: bool,
    optimizer: jopt.Optimizer,
    alignment_weight: float,
    smoothness_weight: float,
    steps_dim_multiple: int,
    smoothing_kernel: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns a tangent vector field and the functional values."""
    arr = jax.lax.stop_gradient(arr)
    arr_smoothed = utils.padded_conv(arr, smoothing_kernel, padding_mode="wrap")

    gx, gy = compute_gradient(arr_smoothed)
    tx0, ty0 = gy, -gx

    # Normalize the gradient if its maximum magnitude exceeds unity.
    magnitude = utils.magnitude(tx0, ty0)
    norm = jnp.maximum(1.0, jnp.amax(magnitude, axis=(-2, -1), keepdims=True))
    tx0 /= norm
    ty0 /= norm

    # If the permittivity is uniform, the vector field `(tx0, ty0)` will be
    # zero and the optimization will yield `nan`. Make a dummy vector field
    # to avoid this.
    permittivity_gradient_is_zero = jnp.all(
        jnp.isclose(tx0, 0.0), axis=(-2, -1), keepdims=True
    ) & jnp.all(jnp.isclose(ty0, 0.0), axis=(-2, -1), keepdims=True)
    tx0 = jnp.where(permittivity_gradient_is_zero, jnp.ones_like(tx0), tx0)
    ty0 = jnp.where(permittivity_gradient_is_zero, jnp.ones_like(ty0), ty0)

    # Remove the average phase of `(tx0, ty0)`, ensuring it is mostly real.
    tx0, ty0 = _remove_mean_phase(tx0, ty0)

    # Obtain initial `(tx, ty)` by taking the real part of `(tx0, ty0)` and
    # normalizing, so that all nonzero elements have magnitude `1`. This
    # ensures that the initial `(tx, ty)` is well aligned with `(tx0, ty0)`.
    tx, ty = normalize_normal(tx0.real, ty0.real)

    # If the Jones field is sought, transform `(tx, ty)` into a Jones field.
    if use_jones:
        tx, ty = normalize_jones(tx, ty)

    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = functools.partial(
        _field_loss,
        tx0=tx0,
        ty0=ty0,
        alignment_weight=alignment_weight,
        smoothness_weight=smoothness_weight,
    )
    return _optimize_tangent_field(
        tx=tx,
        ty=ty,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps_dim_multiple=steps_dim_multiple,
    )


def _optimize_tangent_field(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    optimizer: jopt.Optimizer,
    steps_dim_multiple: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Optimizes `tx` and `ty` by minimizing a functional.

    Args:
        tx: The initial `tx`.
        ty: The initial `ty`.
        loss_fn: The loss function to be minimized.
        optimizer: The optimizer used to minimize the functional.
        steps_dim_multiple: Controls the number of steps in the optimization. The
            number of steps is the product of `steps_dim_multiple` and the dimension
            of the largest of the two trailing axes of `tx`.

    Returns:
        The tangent vector fields and array giving the functional values throughout
        the optimization, `(tx, ty, values)`.
    """

    def _step_fn(step_state, dummy_x):
        del dummy_x
        step, state = step_state
        tx, ty = optimizer.params_fn(state)
        value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(tx, ty)
        conj_grads = jax.tree_util.tree_map(jnp.conj, grads)
        state = optimizer.update_fn(step, conj_grads, state)
        state = _clip_magnitude_state(state, max_magnitude=1.0)
        return (step + 1, state), value

    num_steps = steps_dim_multiple * max(tx.shape[-2:])
    state = optimizer.init_fn((tx, ty))
    (_, state), values = jax.lax.scan(_step_fn, (0, state), xs=None, length=num_steps)
    tx, ty = optimizer.params_fn(state)
    return tx, ty, values


def _clip_magnitude_state(
    state: jopt.OptimizerState,
    max_magnitude: float,
) -> jopt.OptimizerState:
    """Extracts parameters from `state`, clips their magnitude, and repacks the state."""
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) % 2 == 0
    idx_tx = 0
    idx_ty = len(leaves) // 2
    tx = leaves[idx_tx]
    ty = leaves[idx_ty]
    assert tx.shape == ty.shape
    tx, ty = _clip_magnitude(tx, ty, max_magnitude)
    leaves = list(leaves)
    leaves[idx_tx] = tx
    leaves[idx_ty] = ty
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _field_loss(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    tx0: jnp.ndarray,
    ty0: jnp.ndarray,
    alignment_weight: float,
    smoothness_weight: float,
) -> jnp.ndarray:
    """Returns the tangent loss.

    The tangent loss is minimized when `(tx, ty)` is aligned with
    `(tx0, ty0)` and is smooth.

    Args:
        tx: The x-component of the field to be optimized.
        ty: The y-component of the field to be optimized.
        tx0: The x-component of the target field.
        ty0: The y-component of the target field.
        alignment_weight: Determines the weight of term rewarding alignement of
            `(tx, ty)` with `(tx0, ty0)`.
        smoothness_weight: Determines the weight of the smoothness of `(tx, ty)`.

    Returns:
        The loss value.
    """
    self_alignment_loss = alignment_weight * _self_alignment_loss(tx, ty, tx0, ty0)
    gx = jnp.sum(jnp.abs(jnp.asarray(_forward_difference_gradient(tx))) ** 2)
    gy = jnp.sum(jnp.abs(jnp.asarray(_forward_difference_gradient(ty))) ** 2)
    smoothness_loss = smoothness_weight * (gx + gy)
    return self_alignment_loss + smoothness_loss


def _forward_difference_gradient(arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the gradient by forward finite difference."""
    gx = jnp.roll(arr, axis=-2, shift=-1) - arr
    gy = jnp.roll(arr, axis=-1, shift=-1) - arr
    return gx, gy


def _self_alignment_loss(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    tx0: jnp.ndarray,
    ty0: jnp.ndarray,
) -> jnp.ndarray:
    """Returns a loss associated with alignment of `(tx, ty)` and `(tx0, ty0)`.

    Args:
        tx: The x-component of the field to be optimized.
        ty: The y-component of the field to be optimized.
        tx0: The x-component of the target field.
        ty0: The y-component of the target field.

    Returns:
        The loss value.
    """
    alignment_loss = -jnp.abs(tx * jnp.conj(tx0) + ty * jnp.conj(ty0))
    magnitude_loss = jax.nn.relu(utils.magnitude(tx, ty) - 1) ** 2
    return jnp.sum(alignment_loss + magnitude_loss)


def _clip_magnitude(
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    max_magnitude: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Clips `(ax, ay)` to have a maximum magnitude."""
    magnitude = utils.magnitude(ax, ay)
    magnitude_safe = jnp.where(jnp.isclose(magnitude, 0.0), 1.0, magnitude)
    norm_safe = jnp.where(
        magnitude < max_magnitude,
        1.0,
        max_magnitude / magnitude_safe,
    )
    return ax * norm_safe, ay * norm_safe


def _remove_mean_phase(
    ax: jnp.ndarray, ay: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Removes the average phase of `(ax, ay)`."""
    magnitude = utils.magnitude(ax, ay)
    magnitude_squared_safe = jnp.where(jnp.isclose(magnitude, 0.0), 1.0, magnitude**2)
    phase = (
        utils.angle(ax) * jnp.abs(ax) ** 2 + utils.angle(ay) * jnp.abs(ay) ** 2
    ) / magnitude_squared_safe
    mean_phase = jnp.mean(phase, axis=(-2, -1), keepdims=True)
    norm = jnp.exp(1j * mean_phase)
    return ax / norm, ay / norm


# -----------------------------------------------------------------------------
# Library of available vector field generating schemes.
# -----------------------------------------------------------------------------


VectorFn = Callable[
    [jnp.ndarray, basis.LatticeVectors], Tuple[jnp.ndarray, jnp.ndarray]
]

JONES_DIRECT: str = "jones_direct"
JONES: str = "jones"
NORMAL: str = "normal"
POL: str = "pol"

OPTIMIZER = jopt.momentum(step_size=0.2, mass=0.8)
ALIGNMENT_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 1.0
STEPS_DIM_MULTIPLE = 10.0
SMOOTHING_KERNEL = utils.gaussian_kernel(shape=(9, 9), fwhm=3.0)
RESIZE_MAX_DIM = 140
RESIZE_METHOD = jax.image.ResizeMethod.CUBIC


VECTOR_FIELD_SCHEMES: Dict[str, VectorFn] = {
    JONES_DIRECT: functools.partial(
        normalized_vector_field,
        vector_fn=functools.partial(
            tangent_field,
            use_jones=True,  # Directly compute Jones field.
            optimizer=OPTIMIZER,
            alignment_weight=ALIGNMENT_WEIGHT,
            smoothness_weight=SMOOTHNESS_WEIGHT,
            steps_dim_multiple=STEPS_DIM_MULTIPLE,
            smoothing_kernel=SMOOTHING_KERNEL,
        ),
        # The `JONES_DIRECT` scheme directly produces the Jones field, which we
        # just normalize to ensure the elements have a maximum magnitude of `1`.
        normalize_fn=normalize_pol,
        resize_max_dim=RESIZE_MAX_DIM,
        resize_method=RESIZE_METHOD,
    ),
    JONES: functools.partial(
        normalized_vector_field,
        vector_fn=functools.partial(
            tangent_field,
            use_jones=False,  # Obtain Jones field by normalization.
            optimizer=OPTIMIZER,
            alignment_weight=ALIGNMENT_WEIGHT,
            smoothness_weight=SMOOTHNESS_WEIGHT,
            steps_dim_multiple=STEPS_DIM_MULTIPLE,
            smoothing_kernel=SMOOTHING_KERNEL,
        ),
        normalize_fn=normalize_jones,
        resize_max_dim=RESIZE_MAX_DIM,
        resize_method=RESIZE_METHOD,
    ),
    NORMAL: functools.partial(
        normalized_vector_field,
        vector_fn=functools.partial(
            tangent_field,
            use_jones=False,
            optimizer=OPTIMIZER,
            alignment_weight=ALIGNMENT_WEIGHT,
            smoothness_weight=SMOOTHNESS_WEIGHT,
            steps_dim_multiple=STEPS_DIM_MULTIPLE,
            smoothing_kernel=SMOOTHING_KERNEL,
        ),
        normalize_fn=normalize_normal,
        resize_max_dim=RESIZE_MAX_DIM,
        resize_method=RESIZE_METHOD,
    ),
    POL: functools.partial(
        normalized_vector_field,
        vector_fn=functools.partial(
            tangent_field,
            use_jones=False,
            optimizer=OPTIMIZER,
            alignment_weight=ALIGNMENT_WEIGHT,
            smoothness_weight=SMOOTHNESS_WEIGHT,
            steps_dim_multiple=STEPS_DIM_MULTIPLE,
            smoothing_kernel=SMOOTHING_KERNEL,
        ),
        normalize_fn=normalize_pol,
        resize_max_dim=RESIZE_MAX_DIM,
        resize_method=RESIZE_METHOD,
    ),
}
