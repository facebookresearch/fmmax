"""A photonic crystal with Gaussian beam excitation and internal source.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as onp
from skimage import measure  # type: ignore[import]

from fmmax import basis, beams, fields, fmm, scattering, sources

PERMITTIVITY_AMBIENT: complex = (1.0 + 0.0j) ** 2
PERMITTIVITY_SLAB: complex = (1.5 + 0.0j) ** 2
THICKNESS_AMBIENT: float = 2.0
THICKNESS_SLAB: float = 0.8
PITCH: float = 1.0
DIAMETER: float = 0.7
RESOLUTION: float = 0.01
RESOLUTION_FIELDS: float = 0.01
WAVELENGTH: float = 0.63
MULTIPLE_WAVELENGTHS: jnp.ndarray = jnp.asarray([0.62, 0.63, 0.64])
APPROXIMATE_NUM_TERMS: int = 50
BRILLOUIN_GRID_SHAPE: Tuple[int, int] = (9, 9)
WAVELENGTH_AXIS: int = 0


def simulate_crystal_with_internal_source(
    permittivity_ambient: complex = PERMITTIVITY_AMBIENT,
    permittivity_slab: complex = PERMITTIVITY_SLAB,
    thickness_ambient: float = THICKNESS_AMBIENT,
    thickness_slab: float = THICKNESS_SLAB,
    pitch: float = PITCH,
    diameter: float = DIAMETER,
    resolution: float = RESOLUTION,
    resolution_fields: float = RESOLUTION_FIELDS,
    wavelength: float = WAVELENGTH,
    approximate_num_terms: int = APPROXIMATE_NUM_TERMS,
    brillouin_grid_shape: Tuple[int, int] = BRILLOUIN_GRID_SHAPE,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (ex, ey, ez)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (hx, hy, hz)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (x, y, z)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (xy, xz, yz) cross sections
]:
    """Simulates a dipole source inside a photonic crystal slab.

    The crystal has a square unit cell with circular holes, having cross section
    and dipole position as illustrated below. The dipole is located the lower-left
    corner of the unit cell centered in the supercell defined by the Brillouin grid
    shape. The dipole is x-oriented and centered vertically within the photonic
    crystal slab.
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
        (hx, hy, hz), (x, y, z))`. The fields are returned for an xz slice centered
        on the dipole.
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
        fmm.eigensolve_isotropic_media,
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT,
    )

    mask = unit_cell_pattern(pitch, diameter, resolution)
    permittivity_crystal = jnp.where(mask, permittivity_ambient, permittivity_slab)
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
    dipole_x = pitch * brillouin_grid_shape[0] // 2
    dipole_y = pitch * brillouin_grid_shape[1] // 2
    dipole = sources.dirac_delta_source(
        location=jnp.asarray([[dipole_x, dipole_y]]),
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
    # Coordinates where fields are to be evaluated.
    x = jnp.arange(0, pitch * brillouin_grid_shape[0], resolution_fields)
    y = jnp.ones_like(x) * pitch * brillouin_grid_shape[1] // 2
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
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
        layer_znum=[
            int(thickness_ambient_ / resolution_fields),
            int(thickness_slab_ / resolution_fields / 2),
            int(thickness_slab_ / resolution_fields / 2),
            int(thickness_ambient_ / resolution_fields),
        ],
        x=x,
        y=y,
    )

    # Perform the Brillouin zone integration by averaging over the Brillouin zone
    # grid batch axes.
    ex, ey, ez, hx, hy, hz = [
        jnp.mean(field, axis=(0, 1)) for field in (ex, ey, ez, hx, hy, hz)
    ]

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


def simulate_crystal_with_gaussian_beam(
    polar_angle: float = 0.15 * jnp.pi,
    azimuthal_angle: float = 0.0,
    polarization_angle: float = 0.0,
    beam_waist: float = 1.0,
    beam_focus_offset: float = 0.0,
    permittivity_ambient: complex = PERMITTIVITY_AMBIENT,
    permittivity_slab: complex = PERMITTIVITY_SLAB,
    thickness_ambient: float = THICKNESS_AMBIENT,
    thickness_slab: float = THICKNESS_SLAB,
    pitch: float = PITCH,
    diameter: float = DIAMETER,
    resolution: float = RESOLUTION,
    resolution_fields: float = RESOLUTION_FIELDS,
    wavelengths: jnp.ndarray = MULTIPLE_WAVELENGTHS,
    approximate_num_terms: int = APPROXIMATE_NUM_TERMS,
    brillouin_grid_shape: Tuple[int, int] = BRILLOUIN_GRID_SHAPE,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (ex, ey, ez)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (hx, hy, hz)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (x, y, z)
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (xy, xz, yz) cross sections
]:
    """Simulates a "broadband" Gaussian beam incident on photonic crystal slab.

    The crystal has a square unit cell with circular holes as illustrated below.
                     ________________
                    |                |
                    |XX            XX|
                    |XXXX        XXXX|
                    |XXXX        XXXX|
                    |XX            XX|
                    |________________|

    Args:
        polar_angle: The polar angle of the incident beam.
        azimuthal_angle: The azimuthal angle of the incident beam.
        polarization_angle: The angle giving the polarization rotation about the
            propagation axis.
        beam_waist: The Gaussian beam waist.
        beam_focus_offset: The offset of the Gaussian beam focus from the top of the
            photonic crystal slab.
        permittivity_ambient: Permittivity of the region above and below the slab, and
            of the holes in the slab.
        permittivity_slab: Permittivity of the slab.
        thickness_ambient: Thickness of the ambient layers above and below the slab.
        thickness_slab: Thickness of the photonic crystal slab.
        pitch: The unit cell pitch.
        diameter: The diameter of the holes in the photonic crystal.
        resolution: The size of a pixel in permittivity arrays.
        resolution_fields: The size of a pixel in field arrays.
        wavelengths: The wavelengths, of the gaussian beam.
        approximate_num_terms: The number of terms in the Fourier expansion.
        brillouin_grid_shape: The shape of the grid used for Brillouin zone integration.

    Returns:
        The electric and magnetic fields, and the grid coordinates, `((ex, ey, ez),
        (hx, hy, hz), (x, y, z))`. The fields are returned for an xz slice centered
        on the incident beam.
    """
    wavelengths = jnp.expand_dims(jnp.atleast_1d(wavelengths), axis=(1, 2))
    assert wavelengths.ndim == 3

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
    # distributed throughout the first Brillouin zone. We shift the expansion so
    # that it is centered on the direction of the incident beam.
    in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
        brillouin_grid_shape, primitive_lattice_vectors
    )
    in_plane_wavevector += basis.plane_wave_in_plane_wavevector(
        wavelength=jnp.asarray(wavelengths),
        polar_angle=jnp.asarray(polar_angle),
        azimuthal_angle=jnp.asarray(azimuthal_angle),
        permittivity=jnp.asarray(permittivity_ambient),
    )

    assert in_plane_wavevector.shape[0] == wavelengths.size
    assert in_plane_wavevector.shape[1] == brillouin_grid_shape[0]
    assert in_plane_wavevector.shape[2] == brillouin_grid_shape[1]
    assert in_plane_wavevector.shape[-1] == 2
    assert in_plane_wavevector.ndim == 4

    eigensolve = functools.partial(
        fmm.eigensolve_isotropic_media,
        wavelength=jnp.asarray(wavelengths),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT,
    )

    mask = unit_cell_pattern(pitch, diameter, resolution)
    permittivity_crystal = jnp.where(mask, permittivity_ambient, permittivity_slab)
    solve_result_crystal = eigensolve(permittivity=permittivity_crystal)
    solve_result_ambient = eigensolve(
        permittivity=jnp.asarray(permittivity_ambient)[jnp.newaxis, jnp.newaxis]
    )

    s_matrices_interior = scattering.stack_s_matrices_interior(
        layer_solve_results=[
            solve_result_ambient,
            solve_result_crystal,
            solve_result_ambient,
        ],
        layer_thicknesses=[thickness_ambient_, thickness_slab_, thickness_ambient_],
    )

    # Now compute the eigenmode amplitudes for an incident Gaussian beam.
    # This is done by first obtaining the electric and magnetic fields for the
    # beam, and then solving for the eigenmodes.
    # TODO: replace paraxial Gaussian with a more rigorous expression.

    def _paraxial_gaussian_field_fn(x, y, z):
        # Returns the fields of a z-propagating, x-polarized Gaussian beam.
        # See https://en.wikipedia.org/wiki/Gaussian_beam

        # Adjust array dimensions for proper batching
        wavelengths_padded = wavelengths[..., jnp.newaxis, jnp.newaxis]

        k = 2 * jnp.pi / wavelengths_padded
        z_r = (
            jnp.pi * beam_waist**2 * jnp.sqrt(permittivity_ambient) / wavelengths_padded
        )
        w_z = beam_waist * jnp.sqrt(1 + (z / z_r) ** 2)
        r = jnp.sqrt(x**2 + y**2)
        ex = (
            beam_waist
            / w_z
            * jnp.exp(-(r**2) / w_z**2)
            * jnp.exp(
                1j
                * (
                    (k * z)  # Phase
                    + k * r**2 / 2 * z / (z**2 + z_r**2)  # Wavefront curvature
                    - jnp.arctan(z / z_r)  # Gouy phase
                )
            )
        )
        ey = jnp.zeros_like(ex)
        ez = jnp.zeros_like(ex)
        hx = jnp.zeros_like(ex)
        hy = ex / jnp.sqrt(permittivity_ambient)
        hz = jnp.zeros_like(ex)
        return (ex, ey, ez), (hx, hy, hz)

    # Solve for the fields of the beam with the desired rotation and shift.
    x, y = basis.unit_cell_coordinates(
        primitive_lattice_vectors=primitive_lattice_vectors,
        shape=permittivity_crystal.shape[-2:],  # type: ignore[arg-type]
        num_unit_cells=brillouin_grid_shape,
    )
    (beam_ex, beam_ey, _), (beam_hx, beam_hy, _) = beams.shifted_rotated_fields(
        field_fn=_paraxial_gaussian_field_fn,
        x=x,
        y=y,
        z=jnp.zeros_like(x),
        beam_origin_x=jnp.amax(x) / 2,
        beam_origin_y=jnp.amax(y) / 2,
        beam_origin_z=thickness_ambient_ - beam_focus_offset,
        polar_angle=jnp.asarray(polar_angle),
        azimuthal_angle=jnp.asarray(azimuthal_angle),
        polarization_angle=jnp.asarray(polarization_angle),
    )

    brillouin_grid_axes = (1, 2)
    # Add an additional axis for the number of sources
    fwd_amplitude, _ = sources.amplitudes_for_fields(
        ex=beam_ex[..., jnp.newaxis],
        ey=beam_ey[..., jnp.newaxis],
        hx=beam_hx[..., jnp.newaxis],
        hy=beam_hy[..., jnp.newaxis],
        layer_solve_result=solve_result_ambient,
        brillouin_grid_axes=brillouin_grid_axes,
    )

    # Compute the fields inside the structure.
    amplitudes_interior = fields.stack_amplitudes_interior(
        s_matrices_interior=s_matrices_interior,
        forward_amplitude_0_start=fwd_amplitude,
        backward_amplitude_N_end=jnp.zeros_like(fwd_amplitude),
    )
    # Coordinates where fields are to be evaluated.
    x = jnp.arange(0, pitch * brillouin_grid_shape[0], resolution_fields)
    y = jnp.ones_like(x) * pitch * brillouin_grid_shape[1] / 2
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=[
            solve_result_ambient,
            solve_result_crystal,
            solve_result_ambient,
        ],
        layer_thicknesses=[
            thickness_ambient_,
            thickness_slab_,
            thickness_ambient_,
        ],
        layer_znum=[
            int(thickness_ambient_ / resolution_fields),
            int(thickness_slab_ / resolution_fields),
            int(thickness_ambient_ / resolution_fields),
        ],
        x=x,
        y=y,
    )

    # Perform the Brillouin zone integration by averaging over the Brillouin zone
    # grid batch axes.
    ex, ey, ez, hx, hy, hz = [
        jnp.mean(field, axis=brillouin_grid_axes) for field in (ex, ey, ez, hx, hy, hz)
    ]

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


def plot_dipole_fields(
    pitch: float = PITCH,
    resolution: float = RESOLUTION,
    resolution_fields: float = RESOLUTION_FIELDS,
    brillouin_grid_shape: Tuple[int, int] = BRILLOUIN_GRID_SHAPE,
    **sim_kwargs,
) -> None:
    """Plots an electric field slice for the crystal with embedded source."""
    sim_kwargs.update(
        {
            "pitch": pitch,
            "brillouin_grid_shape": brillouin_grid_shape,
            "resolution": resolution,
            "resolution_fields": resolution_fields,
        }
    )
    (
        (ex, ey, ez),
        (hx, hy, hz),
        (x, y, z),
        (section_xy, section_xz, section_yz),
    ) = simulate_crystal_with_internal_source(**sim_kwargs)

    xplot, zplot = jnp.meshgrid(x, z, indexing="ij")
    field_plot = ex[:, :, 0].real

    plt.figure(figsize=(float(jnp.amax(xplot)), float(jnp.amax(zplot))), dpi=80)
    ax = plt.subplot(111)
    im = plt.pcolormesh(xplot, zplot, field_plot, shading="nearest", cmap="bwr")

    im.set_clim((-float(jnp.amax(field_plot)), float(jnp.amax(field_plot))))

    contours = measure.find_contours(onp.array(section_xz))
    scale_factor = pitch / resolution
    for c in contours:
        ax.plot(c[:, 0] / scale_factor, c[:, 1] / scale_factor, "k")

    ax.axis("equal")
    ax.axis("off")
    ax.set_ylim(ax.get_ylim()[::-1])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    plt.savefig("crystal_dipole.png", bbox_inches="tight")


def plot_gaussian_fields(
    pitch: float = PITCH,
    resolution: float = RESOLUTION,
    resolution_fields: float = RESOLUTION_FIELDS,
    brillouin_grid_shape: Tuple[int, int] = BRILLOUIN_GRID_SHAPE,
    wavelength_idx: int = 0,
    **sim_kwargs,
) -> None:
    """Plots an electric field slice for the crystal with Gaussian beam."""
    sim_kwargs.update(
        {
            "pitch": pitch,
            "brillouin_grid_shape": brillouin_grid_shape,
            "resolution": resolution,
            "resolution_fields": resolution_fields,
        }
    )
    (
        (ex, ey, ez),
        (hx, hy, hz),
        (x, y, z),
        (section_xy, section_xz, section_yz),
    ) = simulate_crystal_with_gaussian_beam(**sim_kwargs)

    xplot, zplot = jnp.meshgrid(x, z, indexing="ij")
    field_plot = ex[wavelength_idx, :, :, 0].real

    plt.figure(figsize=(float(jnp.amax(xplot)), float(jnp.amax(zplot))), dpi=80)
    ax = plt.subplot(111)
    im = plt.pcolormesh(xplot, zplot, field_plot, shading="nearest", cmap="bwr")

    im.set_clim((-float(jnp.amax(field_plot)), float(jnp.amax(field_plot))))

    contours = measure.find_contours(onp.array(section_xz))
    scale_factor = pitch / resolution
    for c in contours:
        ax.plot(c[:, 0] / scale_factor, c[:, 1] / scale_factor, "k")

    ax.axis("equal")
    ax.axis("off")
    ax.set_ylim(ax.get_ylim()[::-1])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig("crystal_gaussian.png", bbox_inches="tight")


if __name__ == "__main__":
    plot_dipole_fields()
    plot_gaussian_fields()
