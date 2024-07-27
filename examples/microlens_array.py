"""An example which models a plane wave incident on a microlens array.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Callable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt  # type: ignore[import]

from fmmax import basis, fields, fmm, scattering, utils


def simulate_microlens_array(
    permittivity_ambient: complex = (1.0 + 0.0001j) ** 2,
    permittivity_substrate: complex = (1.5 + 0.0001j) ** 2,
    thickness_ambient: float = 8.0,
    thickness_substrate: float = 1.0,
    lens_height: float = 0.6,
    num_lens_layers: int = 10,
    pitch: float = 4.0,
    grid_shape: Tuple[int, int] = (100, 100),
    grid_spacing_fields: float = 0.01,
    wavelength: float = 0.63,
    approximate_num_terms: int = 800,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (ex, ey, ez)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (hx, hy, hz)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (x, y, z)
    Tuple[jnp.ndarray, jnp.ndarray],  # (lens_x, lens_z)
]:
    """Simulates a plane wave incident on a microlens array having quadratic profile.

    Args:
        permittivity_ambient: The permittivity of the ambient.
        permittivity_substrate: The permittivity of the substrate.
        thickness_ambient: The thickness of the ambient above the lens.
        thickness_substrate: The thickness of the substrate layer.
        lens_height: The height of the microlens.
        num_lens_layers: The number of layers in the discretized microlens
            profile.
        pitch: The x- and y-extent of the unit cell.
        grid_shape: The shape of the grid used to represent permittivities and
            permeabilities.
        grid_spacing_fields: The spacing of gridpoints for calculated fields.
        wavelength: The wavelength of for the incident plane wave.
        approximate_num_terms: The approximate number of terms used in the plane
            wave expansion of the fields.

    Returns:
        The electric fields, magnetic fields, coordinates where these are evaluated,
        and the `(x, z)` coordinates of the discretized lens profile.
    """
    primitive_lattice_vectors = basis.LatticeVectors(
        u=pitch * basis.X, v=pitch * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=basis.Truncation.CIRCULAR,
    )

    # Generate the permittivity arrays and thicknesses for the layers that
    # comprise the microlens.
    lens_profile_fn = functools.partial(
        lens_profile,
        height=jnp.asarray(lens_height),
        pitch=jnp.asarray(pitch),
    )
    lens_radii = jnp.linspace(0.1, pitch / 2, num_lens_layers)
    lens_layer_permittivities = [
        utils.interpolate_permittivity(
            density=circle_density(radius=r, pitch=pitch, grid_shape=grid_shape),
            permittivity_solid=jnp.asarray(permittivity_substrate),
            permittivity_void=jnp.asarray(permittivity_ambient),
        )
        for r in lens_radii
    ]
    lens_layer_thicknesses = discrete_profile_layer_thicknesses(
        profile_fn=lens_profile_fn,
        r_values=lens_radii,
        r_min=0.0,
        r_max=pitch / 2,
    )

    # Collect all layer permittivities and thicknesses, and perform the eigensolve.
    layer_permittivities = (
        [jnp.full((1, 1), permittivity_ambient)]
        + lens_layer_permittivities
        + [jnp.full((1, 1), permittivity_substrate)]
    )
    layer_thicknesses = (
        [jnp.asarray(thickness_ambient)]
        + list(lens_layer_thicknesses)
        + [jnp.asarray(thickness_substrate)]
    )

    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=jnp.zeros((2,)),
            permittivity=p,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        for p in layer_permittivities
    ]
    s_matrices_interior = scattering.stack_s_matrices_interior(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=layer_thicknesses,
    )

    # Solve for the eigenmode amplitudes in every layer of the stack, given a
    # plane wave incident from the substrate.
    bwd_amplitude_substrate_end = jnp.zeros((2 * expansion.num_terms, 1), dtype=complex)
    bwd_amplitude_substrate_end = bwd_amplitude_substrate_end.at[0, 0].set(1)
    amplitudes_interior = fields.stack_amplitudes_interior(
        s_matrices_interior=s_matrices_interior,
        forward_amplitude_0_start=jnp.zeros_like(bwd_amplitude_substrate_end),
        backward_amplitude_N_end=bwd_amplitude_substrate_end,
    )

    # Compute the fields for a cross section through the center of a lens.
    x = jnp.arange(0, pitch + grid_spacing_fields, grid_spacing_fields)
    y = jnp.ones_like(x) * pitch / 2
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=layer_solve_results,
        layer_thicknesses=layer_thicknesses,
        layer_znum=[int(jnp.ceil(t / grid_spacing_fields)) for t in layer_thicknesses],
        x=x,
        y=y,
    )

    # Generate the lens coordinates to be used in plotting.
    lens_x, lens_z = lens_coordinates(
        pitch=pitch,
        z_offset=thickness_ambient,
        lens_radii=lens_radii,
        lens_layer_thicknesses=lens_layer_thicknesses,
    )

    return (ex, ey, ez), (hx, hy, hz), (x, y, z), (lens_x, lens_z)


def circle_density(
    radius: float, pitch: float, grid_shape: Tuple[int, int]
) -> jnp.ndarray:
    """Generates the density array for a centered circular feature."""
    x, y = jnp.meshgrid(
        jnp.arange(0, grid_shape[0]) / grid_shape[0] * pitch,
        jnp.arange(0, grid_shape[1]) / grid_shape[1] * pitch,
        indexing="ij",
    )
    r = jnp.sqrt((x - pitch / 2) ** 2 + (y - pitch / 2) ** 2)
    return (r <= radius).astype(float)


def discrete_profile_layer_thicknesses(
    profile_fn: Callable[[jnp.ndarray], jnp.ndarray],
    r_values: jnp.ndarray,
    r_min: float,
    r_max: float,
) -> jnp.ndarray:
    r"""Computes layer thicknesses for a discrete approximation of `profile_fn`.

    The heights are obtained from a computation as illustrated below. Given a
    continuous, monotonically increasing profile (left) defined between r=r_min
    and r=r_max, we compute the layer heights that approximately implement the
    profile with a discrete set of r values.

               continuous profile            discretized profile

          r=0_____r_min_____r_max____>      _________r0__r1__r2___>
            |     |         |              |         |   |  /
            |     o__                      |_________      |
            |         \_                   |     h0  |___
            |            \                 |         h1  |
            |              \               |             |_
          z |               |              |           h2  |
            V               o              V               o

    Args:
        profile_fn: Monotonically increasing function describing the profile.
        r_values: The monotonically-increasing discrete `r` values used in the
            discrete approximation of the profile.
        r_min: The minimum `r` value of the profile.
        r_max: The maximum `r` value of the profile.

    Returns:
        The layer thicknesses.
    """
    r_values = jnp.asarray(r_values)

    r_mid = (r_values[1:] + r_values[:-1]) / 2
    r_mid = jnp.clip(r_mid, r_min, r_max)
    z_mid = profile_fn(r_mid)

    height_interior = z_mid[1:] - z_mid[:-1]
    height_start = z_mid[0] - profile_fn(jnp.asarray(r_min))
    height_end = profile_fn(jnp.asarray(r_max)) - z_mid[-1]

    return jnp.concatenate(
        [
            height_start[jnp.newaxis],
            height_interior,
            height_end[jnp.newaxis],
        ]
    )


def lens_profile(
    r: jnp.ndarray, height: jnp.ndarray, pitch: jnp.ndarray
) -> jnp.ndarray:
    """Return the z-coordinate for a quadratic lens.

    The profile is such that `profile(0) == 0` and `profile(pitch / 2) == height`.

    Args:
        r: The position where the z-coordinate is to be evaluated.
        height: The height of the lens.
        pitch: The pitch of the lens.

    Returns:
        The z-coordinate for the given `r`.
    """
    return (4 * height / pitch**2) * r**2


def lens_coordinates(
    pitch: float,
    z_offset: float,
    lens_radii: jnp.ndarray,
    lens_layer_thicknesses: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Generate the `(x, z)` coordinates on the right side of the lens.
    lens_x_ = [pitch / 2]
    lens_z_ = [z_offset]
    for r, t in zip(lens_radii, lens_layer_thicknesses):
        lens_x_ += [pitch / 2 + r, pitch / 2 + r]
        lens_z_ += [lens_z_[-1], lens_z_[-1] + t]
    lens_x = jnp.asarray(lens_x_)
    lens_z = jnp.asarray(lens_z_)
    # Mirror the `(x, z)` coordinates and concatenate to get the full set
    # of coordinates.
    lens_x = jnp.concatenate([pitch - lens_x[::-1], lens_x])
    lens_z = jnp.concatenate([lens_z[::-1], lens_z])
    return lens_x, lens_z


def plot_microlens_array_fields(**kwargs) -> None:
    """Simulates and plots the fields for a plane wave incident on a microlens."""
    (ex, ey, ez), _, (x, y, z), (lens_x, lens_z) = simulate_microlens_array(**kwargs)

    field_plot = jnp.sqrt(jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2 + jnp.abs(ez) ** 2)
    field_plot = field_plot[:, :, 0]

    xplot, zplot = jnp.meshgrid(x, z, indexing="ij")

    plt.figure(figsize=(float(jnp.amax(xplot)), float(jnp.amax(zplot))), dpi=80)
    ax = plt.subplot(111)
    ax.pcolormesh(xplot, zplot, field_plot, shading="nearest", cmap="viridis")

    ax.plot(lens_x, lens_z, "w")

    ax.axis("equal")
    ax.axis("off")
    ax.set_ylim(ax.get_ylim()[::-1])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    plt.savefig("metal_dipole.png", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    plot_microlens_array_fields()
