"""Functions related to beam profiles to be used as sources.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Functions related to arbitrary reorientation of beams.
# -----------------------------------------------------------------------------

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
        polar_angle:
        azimuthal_angle:
        polarization_angle:

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
                ux * uz * (1 - cos_theta_p) + uy * sin_theta_p
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