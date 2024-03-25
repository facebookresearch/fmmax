"""Functions related to perfectly matched layers.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
from typing import Tuple

import jax.numpy as jnp
from jax import tree_util


@dataclasses.dataclass
class PMLParams:
    """Stores parameters that define perfectly matched layers.

    Attributes:
        num_x: The number of grid points occupied by the PML, in the x-direction.
        num_y: The number of grid points occupied by the PML, in the y-direction.
        a_max: The strength parameter for uniaxial PML.
        p: The exponent parameter for uniaxial PML.
        sigma_max: The conductivity parameter for uniaxial PML.
    """

    num_x: int
    num_y: int
    a_max: float = 4.0
    p: float = 4.0
    sigma_max: float = 1.0


def apply_uniaxial_pml(
    permittivity: jnp.ndarray,
    pml_params: PMLParams,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Generate the permittivity and permeability tensor elements for uniaxial pml.

    The PML assumes that the unit cell has primitive lattice vectors u and v
    which are parallel to x and y axes, respectively.

    This function is appropriate for isotropic nonmagnetic media, but the
    permittivities and permeabilities generated are anisotropic.

    Args:
        permittivity: isotropic permittivity
        pml_params: The parameters defining the perfectly matched layer dimensions
            and absorption characteristics.

    Returns:
        The permittivity and permeability tensor elements,
        `((permittivity_xx, permittivity_xy, permittivity_yx, permittivity_yy, permittivity_zz),
          (permeability_xx, permeability_xy, permeability_yx, permeability_yy, permeability_zz))`.
    """
    permittivity = _crop_and_edge_pad_pml_region(
        permittivity, widths=(pml_params.num_x, pml_params.num_y)
    )

    dx, dy = _normalized_distance_into_pml(
        shape=permittivity.shape[-2:],  # type: ignore[arg-type]
        widths=(pml_params.num_x, pml_params.num_y),
    )

    sx = (1 + pml_params.a_max * dx**pml_params.p) * (
        1 + 1j * pml_params.sigma_max * jnp.sin(jnp.pi / 2 * dx) ** 2
    )
    sy = (1 + pml_params.a_max * dy**pml_params.p) * (
        1 + 1j * pml_params.sigma_max * jnp.sin(jnp.pi / 2 * dy) ** 2
    )
    sz = 1

    permittivity_xx = sy * sz / sx * permittivity
    permittivity_yy = sx * sz / sy * permittivity
    permittivity_zz = sx * sy / sz * permittivity
    permittivity_xy = jnp.zeros_like(permittivity)
    permittivity_yx = jnp.zeros_like(permittivity)

    permeability_xx = sy * sz / sx * jnp.ones_like(permittivity)
    permeability_yy = sx * sz / sy * jnp.ones_like(permittivity)
    permeability_zz = sx * sy / sz * jnp.ones_like(permittivity)
    permeability_xy = jnp.zeros_like(permittivity)
    permeability_yx = jnp.zeros_like(permittivity)

    return (
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
    ), (
        permeability_xx,
        permeability_xy,
        permeability_yx,
        permeability_yy,
        permeability_zz,
    )


def _crop_and_edge_pad_pml_region(
    permittivity: jnp.ndarray,
    widths: Tuple[int, int],
) -> jnp.ndarray:
    """Crops the trailing dimensions of `permittivity` and applies edge padding."""
    i_width, j_width = widths
    if (i_width * 2, j_width * 2) >= permittivity.shape[-2:]:
        raise ValueError(
            f"`widths` {widths} are incompatible with permittivity shape "
            f"{permittivity.shape}."
        )

    arr_cropped = permittivity[
        ...,
        i_width : permittivity.shape[-2] - i_width,
        j_width : permittivity.shape[-1] - j_width,
    ]

    pad_width = ((0, 0),) * (permittivity.ndim - 2) + (
        (i_width, i_width),
        (j_width, j_width),
    )
    return jnp.pad(arr_cropped, pad_width, mode="edge")


def _normalized_distance_into_pml(
    shape: Tuple[int, int],
    widths: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the distance into the PML layer, in terms of array elements."""
    i, j = jnp.meshgrid(
        jnp.arange(shape[-2]),
        jnp.arange(shape[-1]),
        indexing="ij",
    )
    i_width, j_width = widths

    i_distance = jnp.maximum(i_width - i, 0)
    i_distance = jnp.maximum(i_distance, i - (shape[-2] - i_width - 1))

    j_distance = jnp.maximum(j_width - j, 0)
    j_distance = jnp.maximum(j_distance, j - (shape[-1] - j_width - 1))

    i_scale = jnp.where(widths[0] == 0, 1, 1 / jnp.where(widths[0] == 0, 1, widths[0]))
    j_scale = jnp.where(widths[1] == 0, 1, 1 / jnp.where(widths[1] == 0, 1, widths[1]))
    return i_distance * i_scale, j_distance * j_scale


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


tree_util.register_pytree_node(
    PMLParams,
    lambda p: ((), (p.num_x, p.num_y, p.a_max, p.p, p.sigma_max)),
    lambda values, _: PMLParams(*values),
)
