"""Functions related to perfectly matched layers.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax.numpy as jnp

from fmmax import basis


def apply_uniaxial_pml(
    permittivity: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    width_u: int,
    width_v: int,
    a_max: float = 4.0,
    p: float = 4.0,
    sigma_max: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate the anisotropic permittivity tensor elements implementing a uniaxial pml.

    Args:
        permittivity:
        primitive_lattice_vectors:
        width_u: Number of elements in the permittivity array to be used for the
            perfectly matched layer, in the `u` direction.
        width_v: Number of elements for the perfectly matched layer, `v` direction.

    Returns:
    """
    # Remove the permittivity in regions within the PML, and replace these with whatever
    # values exist just outside the border of the PML region.
    permittivity = _crop_and_edge_pad_pml_region(
        permittivity, widths=(width_u, width_v)
    )

    du, dv = _normalized_distance_into_pml(
        permittivity.shape, widths=(width_u, width_v)
    )

    su = (1 + a_max * du**p) * (1 + 1j * sigma_max * jnp.sin(jnp.pi / 2 * du) ** 2)
    sv = (1 + a_max * dv**p) * (1 + 1j * sigma_max * jnp.sin(jnp.pi / 2 * dv) ** 2)
    sz = 1

    permittivity_uu = sv * sz / su * permittivity
    permittivity_vv = su * sz / sv * permittivity
    permittivity_zz = su * sv / sz * permittivity

    permittivity_xx = permittivity_uu
    permittivity_xy = jnp.zeros_like(permittivity)
    permittivity_yx = jnp.zeros_like(permittivity)
    permittivity_yy = permittivity_vv

    return (
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
    )


def _crop_and_edge_pad_pml_region(
    arr: jnp.ndarray,
    widths: Tuple[int, int],
) -> jnp.ndarray:
    """Crops the trailing dimensions of `arr` and applies edge padding."""
    i_width, j_width = widths
    if (i_width * 2, j_width * 2) >= arr.shape[-2:]:
        raise ValueError()

    arr_cropped = arr[
        ..., i_width : arr.shape[-2] - i_width, j_width : arr.shape[-1] - j_width
    ]

    pad_width = ((0, 0),) * (arr.ndim - 2) + ((i_width, i_width), (j_width, j_width))
    return jnp.pad(arr_cropped, pad_width, mode="edge")


def _normalized_distance_into_pml(
    shape: Tuple[int, int],
    widths: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """"""
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
