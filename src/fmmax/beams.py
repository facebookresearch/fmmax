"""Functions related to beam profiles to be used as sources.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Callable, Tuple

import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Functions related to arbitrary reorientation of beams.
# -----------------------------------------------------------------------------

Fields = Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Ex, Ey, Ez
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Hx, Hy, Hz
]


def shifted_rotated_fields(
    field_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Fields],
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    beam_origin_x: jnp.ndarray,
    beam_origin_y: jnp.ndarray,
    beam_origin_z: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    polarization_angle: jnp.ndarray,
) -> Fields:
    """Computes the fields on a rotated coordinate system.

    Given `fields_fn(xf, yf, zf) -> (exf, eyf, ezf), (hxf, hyf, hzf)` which
    returns the fields _in the field coordinate system_, returns the fields
    at coordinates `(x, y, z)`, which are rotated from `(xf, yf, zf)`, by
    the specified `polar_angle`, `azimuthal_angle`, and `polarization_angle`.

    A beam propagating in the `zf` direction, polarized in the `xf` direction
    will be propagating in the direction specified by `polar_angle` and
    `azimuthal_angle`, with polarization rotated about the propagation
    direction by `polarization_angle`.

    Args:
        field_fn: Function which returns the fields in the field coordinate
            system. The fields should be for a beam propagating in the zf
            direction, i.e. in the z-direction of the beam coordinate system.
        x: x-coordinates of the desired output fields.
        y: y-coordinates of the desired output fields.
        z: z-coordinates of the desired output fields.
        beam_origin_x: The x-origin of the beam coordinate system in the
            `(x, y, z)` unit system.
        beam_origin_y: The y-origin of the beam coordinate system.
        beam_origin_z: The z-origin of the beam coordinate system.
        polar_angle: The rotation angle about the y-axis.
        azimuthal_angle: The rotation angle about the z-axis.
        polarization_angle: The rotation angle about the propagation axis.

    Returns:
        The fields `((ex, ey, ez), (hx, hy, hz))` at the specified coordinates.
    """
    mat = rotation_matrix(polar_angle, azimuthal_angle, polarization_angle)
    mat = jnp.expand_dims(mat, range(x.ndim))

    # Solve for the `(xf, yf, zf)` locations in the field coordinate system
    # which, when rotated as specified, give us the locations `(x, y, z)`.
    assert x.shape == y.shape == z.shape
    coords = jnp.stack([x, y, z], axis=-1)
    rotated_coords = jnp.linalg.solve(mat, coords[..., jnp.newaxis])[..., 0]
    rotated_coords = jnp.split(rotated_coords, 3, axis=-1)
    xf, yf, zf = [jnp.squeeze(r, axis=-1) for r in rotated_coords]

    # Solve for the rotated origin.
    origin = jnp.stack([beam_origin_x, beam_origin_y, beam_origin_z], axis=-1)
    origin = jnp.expand_dims(origin, range(0, mat.ndim - 2))
    rotated_origin = jnp.linalg.solve(mat, origin[..., jnp.newaxis])[..., 0]
    assert rotated_origin.size == 3
    rotated_origin = jnp.split(rotated_origin, 3, axis=-1)
    xf0, yf0, zf0 = [jnp.squeeze(r) for r in rotated_origin]

    # Compute the fields on the rotated, shifted coordinate system.
    (exr, eyr, ezr), (hxr, hyr, hzr) = field_fn(xf - xf0, yf - yf0, zf - zf0)

    rotated_efield = jnp.stack((exr, eyr, ezr), axis=-1)
    rotated_hfield = jnp.stack((hxr, hyr, hzr), axis=-1)

    # Rotate the fields back onto the original coordinate system.
    efield = mat @ rotated_efield[..., jnp.newaxis]
    ex, ey, ez = jnp.split(efield, 3, axis=-2)
    ex = jnp.squeeze(ex, axis=(-2, -1))
    ey = jnp.squeeze(ey, axis=(-2, -1))
    ez = jnp.squeeze(ez, axis=(-2, -1))

    hfield = mat @ rotated_hfield[..., jnp.newaxis]
    hx, hy, hz = jnp.split(hfield, 3, axis=-2)
    hx = jnp.squeeze(hx, axis=(-2, -1))
    hy = jnp.squeeze(hy, axis=(-2, -1))
    hz = jnp.squeeze(hz, axis=(-2, -1))

    return (ex, ey, ez), (hx, hy, hz)


def rotation_matrix(
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    polarization_angle: jnp.ndarray,
) -> jnp.ndarray:
    """Computes a rotation matrix that arbitrarily reorients a field.

    The rotation operations consist of,
      - rotation about the y-axis by `polar_angle`
      - rotation about the z-axis by `azimuthal_angle`
      - rotation about propagation axis by `polarization_angle`, where
        the propagation axis is found by applying the first two rotations
        about the y- and z-axis.

    Args:
        polar_angle: The rotation angle about the y-axis.
        azimuthal_angle: The rotation angle about the z-axis.
        polarization_angle: The rotation angle about the propagation axis.

    Returns:
        The rotation matrix.
    """
    # Matrix that rotates around the y-axis by `polar_angle`.
    rotation_y_matrix = jnp.asarray(
        [
            [jnp.cos(polar_angle), 0.0, jnp.sin(polar_angle)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(polar_angle), 0.0, jnp.cos(polar_angle)],
        ]
    )

    # Matrix that rotates around the z-axis by `azimuthal_angle`.
    rotation_z_matrix = jnp.asarray(
        [
            [jnp.cos(azimuthal_angle), -jnp.sin(azimuthal_angle), 0.0],
            [jnp.sin(azimuthal_angle), jnp.cos(azimuthal_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Matrix that rotates around the axis defined by the specified polar and
    # azimuthal angle. The unit vector along the axis is `(ux, uy, uz)`.
    ux = jnp.cos(azimuthal_angle) * jnp.sin(polar_angle)
    uy = jnp.sin(azimuthal_angle) * jnp.sin(polar_angle)
    uz = jnp.cos(polar_angle)
    cos_theta_p = jnp.cos(polarization_angle)
    sin_theta_p = jnp.sin(polarization_angle)
    rotation_p_matrix = jnp.asarray(
        [
            [
                cos_theta_p + ux**2 * (1 - cos_theta_p),
                ux * uy * (1 - cos_theta_p) - uz * sin_theta_p,
                ux * uz * (1 - cos_theta_p) + uy * sin_theta_p,
            ],
            [
                uy * ux * (1 - cos_theta_p) + uz * sin_theta_p,
                cos_theta_p + uy**2 * (1 - cos_theta_p),
                uy * uz * (1 - cos_theta_p) - ux * sin_theta_p,
            ],
            [
                uz * ux * (1 - cos_theta_p) - uy * sin_theta_p,
                uz * uy * (1 - cos_theta_p) + ux * sin_theta_p,
                cos_theta_p + uz**2 * (1 - cos_theta_p),
            ],
        ]
    )
    return rotation_p_matrix @ rotation_z_matrix @ rotation_y_matrix
