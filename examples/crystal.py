"""A photonic crystal with Gaussian beam excitation and internal source.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Tuple

import jax.numpy as jnp

from fmmax import basis, fields, fmm, layer, scattering, sources


def simulate_crystal_with_internal_source(
    permittivity_ambient: complex = (1.0 + 0.0j) ** 2,
    permittivity_slab: complex = (1.5 + 0.0j) ** 2,
    thickness_ambient: float = 2.0,
    thickness_slab: float = 0.4,
    pitch: float = 1.0,
    diameter: float = 0.7,
    resolution: float = 0.01,
    resolution_fields: float = 0.05,
    wavelength: float = 0.63,
    approximate_num_terms: int = 50,
    brillouin_grid_shape: Tuple[int, int] = (5, 5),
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (ex, ey, ez)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (hx, hy, hz)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (x, y, z)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (xy, xz, yz) cross sections
]:
    """Simulates a dipole source inside a photonic crystal slab.

    The crystal has a square unit cell with circular holes, having cross section
    and dipole position as illustrated below. The dipole is located at (0, 0), is
    x-oriented and centered vertically within the photonic crystal slab.
                     ________________
                    |                |
                    |XX            XX|
                    |XXXX        XXXX|
                    |XXXX        XXXX|
                    |XX            XX|
        x-dipole -> o________________|

    Args:
        permittivity_ambient: Permittivity of the region above and below the slab, and
            of the holes in the slab.
        permittivity_slab: Permittivity of the slab.
        thickness_ambient: Thickness of the ambient layers above and below the slab.
        thickness_slab: Thickness of the photonic crystal slab.
        pitch: The unit cell pitch.
        diameter: The diameter of the holes in the photonic crystal.
        resolution: The size of a pixel in permittivity arrays.
        resolution_fields: The size of a pixel in field arrays.
        wavelength: The wavelength, of the dipole emission.
        approximate_num_terms: The number of terms in the Fourier expansion.
        brillouin_grid_shape: The shape of the grid used for Brillouin zone integration.

    Returns:
        The electric and magnetic fields, and the grid coordinates, `((ex, ey, ez),
        (hx, hy, hz), (x,y, z))`.
    """
    thickness_ambient_ = jnp.asarray(thickness_ambient)
    thickness_slab_ = jnp.asarray(thickness_slab)
    del thickness_ambient, thickness_slab

    primitive_lattice_vectors = basis.LatticeVectors(
        u=pitch * basis.X, v=pitch * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=basis.Truncation.CIRCULAR,
    )

    # Brillouin zone integration creates a batch of in-plane wavevectors which are
    # distributed throughout the first Brillouin zone.
    in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
        brillouin_grid_shape, primitive_lattice_vectors
    )
    assert in_plane_wavevector.shape[-1] == 2
    assert in_plane_wavevector.ndim == 3

    eigensolve = functools.partial(
        layer.eigensolve_isotropic_media,
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT,
    )

    mask = unit_cell_pattern(pitch, diameter, resolution)
    permittivity_crystal = jnp.where(mask, permittivity_slab, permittivity_slab)
    solve_result_crystal = eigensolve(permittivity=permittivity_crystal)
    solve_result_ambient = eigensolve(
        permittivity=jnp.asarray(permittivity_ambient)[jnp.newaxis, jnp.newaxis]
    )

    # First, we model a dipole inside the photonic crystal. For this, we must break
    # the stack into two, and compute scattering matrices for the stacks above and
    # below the plane containing the dipole. Since we want to visualize fields, we
    # also need the interior scattering matrices.
    s_matrices_interior_before_source = scattering.stack_s_matrices_interior(
        layer_solve_results=[solve_result_ambient, solve_result_crystal],
        layer_thicknesses=[thickness_ambient_, thickness_slab_ / 2],
    )
    s_matrices_interior_after_source = scattering.stack_s_matrices_interior(
        layer_solve_results=[solve_result_crystal, solve_result_ambient],
        layer_thicknesses=[thickness_slab_ / 2, thickness_ambient_],
    )
    # Extract the scattering matrices relating fields at the two ends of each substack.
    s_matrix_before_source = s_matrices_interior_before_source[-1][0]
    s_matrix_after_source = s_matrices_interior_after_source[-1][0]

    # Generate the Fourier representation of a point dipole.
    dipole = sources.dirac_delta_source(
        location=jnp.asarray([[0, 0]]),
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
        jx=dipole,
        jy=jnp.zeros_like(dipole),
        jz=jnp.zeros_like(dipole),
        s_matrix_before_source=s_matrix_before_source,
        s_matrix_after_source=s_matrix_after_source,
    )

    # Compute the fields inside the structure.
    amplitudes_interior = fields.stack_amplitudes_interior_with_source(
        s_matrices_interior_before_source=s_matrices_interior_before_source,
        s_matrices_interior_after_source=s_matrices_interior_after_source,
        backward_amplitude_before_end=bwd_amplitude_before_end,
        forward_amplitude_after_start=fwd_amplitude_after_start,
    )
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_auto_grid(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=[
            solve_result_ambient,
            solve_result_crystal,
            solve_result_crystal,
            solve_result_ambient,
        ],
        layer_thicknesses=[
            thickness_ambient_,
            thickness_slab_ / 2,
            thickness_slab_ / 2,
            thickness_ambient_,
        ],
        resolution=resolution_fields,
        num_unit_cells=brillouin_grid_shape,
    )

    # Perform the Brillouin zone integration by averaging over the Brillouin zone
    # grid batch axes.
    ex = jnp.mean(ex, axis=(0, 1))
    ey = jnp.mean(ey, axis=(0, 1))
    ez = jnp.mean(ez, axis=(0, 1))
    hx = jnp.mean(hx, axis=(0, 1))
    hy = jnp.mean(hy, axis=(0, 1))
    hz = jnp.mean(hz, axis=(0, 1))

    # Compute some cross sections for visualizing the structure.
    section_xy, section_xz, section_yz = crystal_cross_sections(
        thickness_ambient=float(thickness_ambient_),
        thickness_slab=float(thickness_slab_),
        pitch=pitch,
        diameter=diameter,
        resolution=resolution,
        num_unit_cells=brillouin_grid_shape,
    )
    return (ex, ey, ez), (hx, hy, hz), (x, y, z), (section_xy, section_xz, section_yz)


def unit_cell_pattern(
    pitch: float,
    diameter: float,
    resolution: float,
) -> jnp.ndarray:
    """Defines the pattern of the photonic crystal."""
    x, y = jnp.meshgrid(
        jnp.arange(0, pitch, resolution),
        jnp.arange(0, pitch, resolution),
        indexing="ij",
    )
    return (jnp.sqrt((x - pitch / 2) ** 2 + y**2) < diameter / 2) | (
        jnp.sqrt((x - pitch / 2) ** 2 + (y - pitch) ** 2) < diameter / 2
    )


def crystal_cross_sections(
    thickness_ambient: float,
    thickness_slab: float,
    pitch: float,
    diameter: float,
    resolution: float,
    num_unit_cells: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes cross sections of the photonic crystal structure."""
    mask = unit_cell_pattern(pitch, diameter, resolution)

    xy_section = jnp.tile(mask, num_unit_cells)

    xz_slab = mask[:, 0]
    xz_section = jnp.stack(
        (
            [jnp.ones_like(xz_slab)] * int(thickness_ambient / resolution)
            + [xz_slab] * int(thickness_slab / resolution)
            + [jnp.ones_like(xz_slab)] * int(thickness_ambient / resolution)
        ),
        axis=-1,
    )
    xz_section = jnp.tile(xz_section, (num_unit_cells[0], 1))

    yz_slab = mask[0, :]
    yz_section = jnp.stack(
        (
            [jnp.ones_like(yz_slab)] * int(thickness_ambient / resolution)
            + [yz_slab] * int(thickness_slab / resolution)
            + [jnp.ones_like(yz_slab)] * int(thickness_ambient / resolution)
        ),
        axis=-1,
    )
    yz_section = jnp.tile(yz_section, (num_unit_cells[1], 1))

    return xy_section, xz_section, yz_section
