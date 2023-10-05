"""Simulates a dipole above a metal plane, with perfectly matched layers.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt  # type: ignore[import]

from fmmax import basis, fields, fmm, pml, scattering, sources


def simulate_metal_dipole(
    permittivity_ambient: complex = (1.0 + 0.0j),
    permittivity_metal: complex = (-7.632 + 0.731j),
    thickness_dipole_metal_gap: float = 1.0,
    thickness_ambient: float = 2.0,
    thickness_metal: float = 0.1,
    pitch: float = 5.0,
    grid_shape: Tuple[int, int] = (400, 400),
    grid_spacing_fields: float = 0.01,
    wavelength: float = 0.63,
    approximate_num_terms: int = 1200,
    pml_params: pml.PMLParams = pml.PMLParams(num_x=50, num_y=50),
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (ex, ey, ez)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (hx, hy, hz)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (x, y, z)
]:
    """Simulates a y-dipole above a metal plane surrounded by absorbing layers.

    The presence of the metal plane will generally modify the radiated power from
    the dipole, and lead to angular dependence in the emission pattern.A cross
    section of the simulation is depicted below.
                 ___________________________
                |xxx|                   |xxx|
                |xxx|                   |xxx|   x: ambient, pml
                |xxx|                   |xxx|   y: metal, pml
                |xxx|        o <-dipole |xxx|   z: metal, no pml
                |xxx|___________________|xxx|
                |yyy|zzzzzzzzzzzzzzzzzzz|yyy|

    Args:
        permittivity_ambient: The permittivity of the ambient.
        permittivity_metal: The permittivity of the metal.
        thickness_dipole_metal_gap: The distance between the dipole and the metal.
        thickness_ambient: The thickness of the ambient above the dipole.
        thickness_metal: The thickness of the metal layer.
        pitch: The x- and y-extent of the unit cell.
        grid_shape: The shape of the grid used to represent permittivities and
            permeabilities.
        grid_spacing_fields: The spacing of gridpoints for calculated fields.
        wavelength: The wavelength of dipole emission.
        approximate_num_terms: The approximate number of terms used in the plane
            wave expansion of the fields.
        pml_params: Parameters specifying the extent and strength of the perfectly
            matched layers.

    Returns:
        The electric fields, magnetic fields, and coordinates where these are evaluated.
    """
    primitive_lattice_vectors = basis.LatticeVectors(
        u=pitch * basis.X, v=pitch * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=basis.Truncation.CIRCULAR,
    )
    in_plane_wavevector = jnp.zeros((2,))

    # Generate the anisotropic permittivity and permeability that arise from applying
    # a uniaxial PML to the x- and y-boundary. Note that the PML assumes a unit cell
    # with primitive lattice vectors aligned with the x- and y-direction.
    permittivities_ambient_pml, permeabilities_ambient_pml = pml.apply_uniaxial_pml(
        permittivity=jnp.full(grid_shape, permittivity_ambient),
        pml_params=pml_params,
    )
    permittivities_metal_pml, permeabilities_metal_pml = pml.apply_uniaxial_pml(
        permittivity=jnp.full(grid_shape, permittivity_metal),
        pml_params=pml_params,
    )

    eigensolve = functools.partial(
        fmm.eigensolve_general_anisotropic_media,
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT,
        vector_field_source=None,  # Automatically choose the vector field source.
    )
    solve_result_ambient = eigensolve(
        permittivity_xx=permittivities_ambient_pml[0],
        permittivity_xy=permittivities_ambient_pml[1],
        permittivity_yx=permittivities_ambient_pml[2],
        permittivity_yy=permittivities_ambient_pml[3],
        permittivity_zz=permittivities_ambient_pml[4],
        permeability_xx=permeabilities_ambient_pml[0],
        permeability_xy=permeabilities_ambient_pml[1],
        permeability_yx=permeabilities_ambient_pml[2],
        permeability_yy=permeabilities_ambient_pml[3],
        permeability_zz=permeabilities_ambient_pml[4],
    )
    solve_result_metal = eigensolve(
        permittivity_xx=permittivities_metal_pml[0],
        permittivity_xy=permittivities_metal_pml[1],
        permittivity_yx=permittivities_metal_pml[2],
        permittivity_yy=permittivities_metal_pml[3],
        permittivity_zz=permittivities_metal_pml[4],
        permeability_xx=permeabilities_metal_pml[0],
        permeability_xy=permeabilities_metal_pml[1],
        permeability_yx=permeabilities_metal_pml[2],
        permeability_yy=permeabilities_metal_pml[3],
        permeability_zz=permeabilities_metal_pml[4],
    )

    # Compute interior scattering matrices to enable field calculations.
    s_matrices_interior_before_source = scattering.stack_s_matrices_interior(
        layer_solve_results=[solve_result_ambient],
        layer_thicknesses=[jnp.asarray(thickness_ambient)],
    )
    s_matrices_interior_after_source = scattering.stack_s_matrices_interior(
        layer_solve_results=[solve_result_ambient, solve_result_metal],
        layer_thicknesses=[
            jnp.asarray(thickness_dipole_metal_gap),
            jnp.asarray(thickness_metal),
        ],
    )

    # Extract the scattering matrices relating fields at the two ends of each substack.
    s_matrix_before_source = s_matrices_interior_before_source[-1][0]
    s_matrix_after_source = s_matrices_interior_after_source[-1][0]

    # Generate the Fourier representation of a point dipole.
    dipole = sources.dirac_delta_source(
        location=jnp.asarray([[pitch / 2, pitch / 2]]),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )

    # Compute backward eigenmode amplitudes at the end of the layer before the
    # source, and the forward amplitudes the start of the layer after the source.
    (
        _,
        _,
        bwd_amplitude_before_end,
        fwd_amplitude_after_start,
        _,
        _,
    ) = sources.amplitudes_for_source(
        jx=jnp.zeros_like(dipole),
        jy=dipole,
        jz=jnp.zeros_like(dipole),
        s_matrix_before_source=s_matrix_before_source,
        s_matrix_after_source=s_matrix_after_source,
    )

    # Solve for the eigenmode amplitudes in every layer of the stack.
    amplitudes_interior = fields.stack_amplitudes_interior_with_source(
        s_matrices_interior_before_source=s_matrices_interior_before_source,
        s_matrices_interior_after_source=s_matrices_interior_after_source,
        backward_amplitude_before_end=bwd_amplitude_before_end,
        forward_amplitude_after_start=fwd_amplitude_after_start,
    )
    # Coordinates where fields are to be evaluated.
    x = jnp.linspace(0, pitch, grid_shape[0])
    y = jnp.ones_like(x) * pitch / 2
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=[
            solve_result_ambient,
            solve_result_ambient,
            solve_result_metal,
        ],
        layer_thicknesses=[
            jnp.asarray(thickness_ambient),
            jnp.asarray(thickness_dipole_metal_gap),
            jnp.asarray(thickness_metal),
        ],
        layer_znum=[
            int(thickness_ambient / grid_spacing_fields),
            int(thickness_dipole_metal_gap / grid_spacing_fields),
            int(thickness_metal / grid_spacing_fields),
        ],
        x=x,
        y=y,
    )
    return (ex, ey, ez), (hx, hy, hz), (x, y, z)


def plot_metal_dipole_fields(
    clip_percentile: float = 99.8,
    **kwargs,
) -> None:
    """Simulates and plots the fields for a dipole spaced away from a metal plane."""
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = simulate_metal_dipole(**kwargs)

    field_plot = jnp.sqrt(jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2 + jnp.abs(ez) ** 2)
    field_plot = field_plot[:, :, 0]

    xplot, zplot = jnp.meshgrid(x, z, indexing="ij")

    plt.figure(figsize=(float(jnp.amax(xplot)), float(jnp.amax(zplot))), dpi=80)
    ax = plt.subplot(111)
    im = ax.pcolormesh(xplot, zplot, field_plot, shading="nearest", cmap="magma")

    clipval = float(jnp.percentile(field_plot, clip_percentile))
    im.set_clim((0, clipval))

    ax.axis("equal")
    ax.axis("off")
    ax.set_ylim(ax.get_ylim()[::-1])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    plt.savefig("metal_dipole.png", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    plot_metal_dipole_fields()
