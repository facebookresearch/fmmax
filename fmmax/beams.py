"""Functions related to beam profiles to be used as sources.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import vmap

# -----------------------------------------------------------------------------
# Functions related to arbitrary reorientation of beams.
# -----------------------------------------------------------------------------

Fields = Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Ex, Ey, Ez
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Hx, Hy, Hz
]


def rotated_fields(
    field_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Fields],
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
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
            system.
        x: x-coordinates of the desired output fields.
        y: y-coordinates of the desired output fields.
        z: z-coordinates of the desired output fields.
        polar_angle: The rotation angle about the y-axis.
        azimuthal_angle: The rotation angle about the z-axis.
        polarization_angle: The rotation angle about the propagation axis.

    Returns:
        The fields `((ex, ey, ez), (hx, hy, hz))` at the specified coordinates.
    """
    coords = jnp.stack([x, y, z], axis=-1)
    mat = rotation_matrix(polar_angle, azimuthal_angle, polarization_angle)
    mat = jnp.expand_dims(mat, range(x.ndim))

    # Solve for the (x, y, z) locations in the rotated coordinate system.
    rotated_coords = jnp.linalg.solve(mat, coords)
    rotated_coords = jnp.split(rotated_coords, 3, axis=-1)
    rotated_coords = [jnp.squeeze(r, axis=-1) for r in rotated_coords]

    # Compute the fields on the rotated coordinate system.
    (exr, eyr, ezr), (hxr, hyr, hzr) = field_fn(*rotated_coords)

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


# -----------------------------------------------------------------------------
# Functions related to gaussian beam generation
# -----------------------------------------------------------------------------


def _special_norm(x: jnp.ndarray) -> float:
    """Compute an L2-like norm, but without the abs.

    Needed for the complex plane we are working with.
    """
    return jnp.sqrt(jnp.sum(x**2, axis=-1))


def _compute_gaussian_beam_fields(
    r_pts: jnp.ndarray,
    k_vector: jnp.ndarray,
    dipole_center: jnp.ndarray,
    polarization: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the full E and H tensors for a Gaussian beam given a complex dipole."""
    N = r_pts.shape[0]

    k_norm = _special_norm(k_vector)
    R_vector = r_pts - dipole_center[jnp.newaxis, :]
    R = _special_norm(R_vector)

    # Terms that are used multiple times. Also helps with readability.
    ikr = 1j * k_norm * R
    k2r2 = k_norm**2 * R**2
    term1 = 1 + (ikr - 1) / (k2r2)
    term2 = (3 - 3 * ikr - k2r2) / (k2r2 * R * R)
    term3 = k_norm / R * (1j - 1 / (k_norm * R))

    # Outer product term
    RR = R_vector[:, :, jnp.newaxis] * R_vector[:, jnp.newaxis, :]

    # Cross product term
    I3 = jnp.eye(3)
    simple_cross = lambda x, y: jnp.cross(x,y)
    cross_over_matrix = vmap(simple_cross, (None, 0), 0)
    cross_over_pts = vmap(cross_over_matrix, (0, None), 0)
    R_cross_I = cross_over_pts(R_vector, I3)

    # Phasor (spherical wave) term
    exp_fac = jnp.exp(ikr) / (4 * jnp.pi * R)

    # Compute full tensor for E and H
    E_full = exp_fac[:, jnp.newaxis, jnp.newaxis] * (
        term1[:, jnp.newaxis, jnp.newaxis] * I3[jnp.newaxis, :, :]
        + term2[:, jnp.newaxis, jnp.newaxis] * RR
    )
    H_full = (
        exp_fac[:, jnp.newaxis, jnp.newaxis]
        * term3[:, jnp.newaxis, jnp.newaxis]
        * R_cross_I
    )

    # Project onto the polarization vector
    E_pol = jnp.einsum(
        "ijk, ijk -> ij", E_full, polarization[jnp.newaxis, jnp.newaxis, :]
    )
    H_pol = jnp.einsum(
        "ijk, ijk -> ij", H_full, polarization[jnp.newaxis, jnp.newaxis, :]
    )

    return E_pol, H_pol


def _get_incoming_gaussian_beam(r_pts, k_vector, beam_waist, beam_center, polarization):
    """Compute the fields for an "incoming" gaussian beam."""
    k_norm = _special_norm(k_vector)
    k_normalized = k_vector / k_norm
    dipole_imag_center = beam_waist * k_normalized
    # incoming wave has a negative imaginary part
    dipole_center = beam_center - 1j * dipole_imag_center
    return _compute_gaussian_beam_fields(r_pts, k_vector, dipole_center, polarization)


def _get_outgoing_gaussian_beam(r_pts, k_vector, beam_waist, beam_center, polarization):
    """Compute the fields for an "outgoing" gaussian beam."""
    k_norm = _special_norm(k_vector)
    k_normalized = k_vector / k_norm
    dipole_imag_center = beam_waist * k_normalized
    # incoming wave has a positive imaginary part
    dipole_center = beam_center + 1j * dipole_imag_center
    return _compute_gaussian_beam_fields(r_pts, k_vector, dipole_center, polarization)


def get_gaussianbeam_EH(
    r_pts: jnp.ndarray,
    k_vector: jnp.ndarray,
    beam_waist: float,
    beam_center: jnp.ndarray,
    polarization: jnp.ndarray,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Get the field profile of a gaussian beam for a set of points.

    This is a flexible and general approach toward calculating all 6 field
    components of a Gaussian beam with an arbitrary orientation, beam waist, and
    waist location. These fields can then be used to generate the equivalent
    source.

    We compute the field profiles using the complex source point (CSP) method.
    Specifically, we specify a dipole in a complex space. The real part of the
    position vector corresponds to the center of the beam waist in real space.
    The imaginary part of the position vector is the direction of propagation
    (the k-vector). The magnitude of the imaginary part corresponds to the beam
    waist. We simply plug this dipole into the dyadic Greens function to compute
    the full field response at any arbitrary point.

    Importantly, the method generates a "outward propagating" beam for positive
    imaginary components, and an "inward propagating" beam for negative
    imaginary components. The points right at the beam waist start to diverge
    thanks to the ill-conditioning of the Green's function... We can probably
    get around this by using an analytic formulation near this region. But this
    works for now. Because of all this, we need to carefully identify where the
    list of input points are relative to the beam-waist plane and use the
    appropriate complex source dipole.

    Args:
        r_pts: position vectors for each point in the source plane (shape [n,3])
        k_vector: the direction of propagation as a 3D vector. (shape [3,])
        beam_waist: the width of the beam waist beam_center: position vector of
        the beam center (shape [3,])
        polarization: polarization vector (shape
        [3,])

    Returns:
        Ex: x-component of E field
        Ey: y-component of E field
        Ez: z-component of E field
        Hx: x-component of H field
        Hy: y-component of H field
        Hz: z-component of H field
    """

    # Normalize the polarization vector
    polarization = polarization / jnp.linalg.norm(polarization)

    # Get both incoming and outgoing beams
    E_incoming, H_incoming = _get_incoming_gaussian_beam(
        r_pts, k_vector, beam_waist, beam_center, polarization
    )
    E_outgoing, H_outgoing = _get_outgoing_gaussian_beam(
        r_pts, k_vector, beam_waist, beam_center, polarization
    )

    # Determine which beam to use.
    onplane = jnp.einsum(
        "ij, ij -> i", k_vector[jnp.newaxis, :], (r_pts - beam_center[jnp.newaxis, :])
    )
    E = jnp.where(
        onplane[:, jnp.newaxis] == 0.0,
        (E_outgoing - E_incoming),
        jnp.where(onplane[:, jnp.newaxis] < 0.0, E_incoming, E_outgoing),
    )
    H = jnp.where(
        onplane[:, jnp.newaxis] == 0.0,
        (H_outgoing - H_incoming),
        jnp.where(onplane[:, jnp.newaxis] < 0.0, H_incoming, H_outgoing),
    )

    # Clean up
    Ex = E[:, 0]
    Ey = E[:, 1]
    Ez = E[:, 2]
    Hx = H[:, 0]
    Hy = H[:, 1]
    Hz = H[:, 2]

    return Ex, Ey, Ez, Hx, Hy, Hz
