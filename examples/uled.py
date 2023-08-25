"""An example which simulates a micro-LED (uLED) with an internal source.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import time
from typing import Sequence, Tuple

import jax.numpy as jnp

from fmmax import basis, fields, fmm, scattering, sources


def simulate_uled(
    permittivity_ambient: complex = (1.0 + 0.0j),
    permittivity_passivation: complex = (1.5 + 0.0j) ** 2,
    permittivity_epi: complex = (3.0 + 0.0j) ** 2,
    permittivity_metal: complex = (0.2 + 3.3j) ** 2,
    thickness_ambient: float = 0.0,
    thickness_epi_top: float = 500.0,
    thickness_epi_bottom: float = 500.0,
    thickness_passivation: float = 100.0,
    thickness_metal: float = 100.0,
    pitch: float = 1400.0,
    epi_diameter: float = 1000.0,
    resolution: float = 1.0,
    resolution_fields: float = 10.0,
    wavelength: float = 620.0,
    dipole_y_offset: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
    dipole_fwhm: float = 20.0,
    approximate_num_terms: int = 1200,
    truncation: basis.Truncation = basis.Truncation.CIRCULAR,
    formulation: fmm.Formulation = fmm.Formulation.POL,
    brillouin_grid_shape: Tuple[int, int] = (1, 1),
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Computes the extraction efficiency and fields for a uLED.

    The uLED consists of an epi (semiconductor) region, surrounded by a
    passivation dielectric and a metal. The structure is circularly
    symmetric, and the diameter of the epi region is `epi_diameter`.

    Dipoles are located in a plane partway through the epi region.
    Each dipole is located at `(x, y) = (0, offset)` for each of the
    offsets in `dipole_y_offset`. At each location, we model dipoles
    oriented in x-, y-, and z-directions, and each dipole has a spatial
    profile that is Gaussian with full-width at half-max given by
    `dipole_fwhm`.

    A cross section depiction of the structure is shown below.
                 _______________________________________
                |    |  |   epi                 |  |    |
                |    |  |                       |  |    |
                |    |  | - - - - - X - X - X - |  |    |
                |    |  |            dipoles    |  |    |
                |    |  |_______________________|  |    |
                |    |___passivation_______________|    |
                |___metal_______________________________|

    The default calculation models a periodic array of dipoles in a
    periodic array of uLEDs. To model an isolated dipole in a periodic
    array of uLEDs, a `brillouin_grid_shape` larger than the default
    `(1, 1)` can be specified, in which case the calculation is done
    for a grid of k-points in the first Brillouin zone. These can then
    be used to carry out Brillouin zone integration [2022 Lopez-Fraguas].

    [2022 Lopez-Fraguas] E. Lopez-Fraguas et al., "Tripling the light
        extraction efficiency of a deep ultraviolet LED using a
        nanostructured p-contact"
        https://www.nature.com/articles/s41598-022-15499-7

    Args:
        permittivity_ambient: The permittivity of the ambient.
        permittivity_passivation: The permittiity of the passivation.
        permittivity_epi: The permittiity of the epi material.
        permittivity_metal: The permittiity of the metal.
        thickness_ambient: The thickness of the ambient above the uLED.
        thickness_epi_top: The thickness of the epi layer above the source.
        thickness_epi_bottom: The thickness of the epi layer below the source.
        thickness_passivation: The thickness of the passivation layer.
        thickness_metal: The thickness of the metal below the passivation.
        pitch: The pitch of the uLED, i.e. the width.
        epi_diameter: The diameter of the epi region.
        resolution: The resolution of the real-space grid for permittivities.
        resolution_fields: The resolution of the real-space grid for fields.
        wavelength: The wavelength of the calculation.
        dipole_y_offset: Sequence of offsets for the dipole along the y axis.
            The dipole position is at `epi_diameter * dipole_y_offset`.
        dipole_fwhm: The dipole spatial full-width at half-maximum amplitude.
        approximate_num_terms: The approximate number of terms used in the
            plane wave expansion of the fields.
        truncation: Determines how the Fourier expansion is truncated.
        formulation: Specifies the formulation to be used.
        brillouin_grid_shape: The shape of the grid of k-points in the first
            Brillouin zone for which the calculation is done.

    Returns:
        extraction_efficiency: The computed extraction efficiency.
        total_emitted_power: The total power emitted by the dipole.
        efields: The electric fields in the structure, with shape `(3,)
            + brillouin_grid_shape + (xnum, ynum, znum, num_dipoles)`.
        hfields: The magnetic fields with shape matching `efields`.
        grid: The coordinates `(x, y, z)` for the fields.
    """
    (
        permittivities_before_source,
        permittivities_after_source,
        thicknesses_before_source,
        thicknesses_after_source,
        primitive_lattice_vectors,
    ) = uled_structure(
        permittivity_ambient=permittivity_ambient,
        permittivity_passivation=permittivity_passivation,
        permittivity_epi=permittivity_epi,
        permittivity_metal=permittivity_metal,
        thickness_ambient=thickness_ambient,
        thickness_epi_top=thickness_epi_top,
        thickness_epi_bottom=thickness_epi_bottom,
        thickness_passivation=thickness_passivation,
        thickness_metal=thickness_metal,
        pitch=pitch,
        epi_diameter=epi_diameter,
        resolution=resolution,
    )

    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=truncation,
    )

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
        formulation=formulation,
    )

    # Perform the layer eigensolve for each layer in the stack. Use the fact that
    # the first layer after the source is identical to the layer just before the
    # source in order to avoid one eigensolve.
    layer_solve_results_before_source = tuple(
        [eigensolve(permittivity=p) for p in permittivities_before_source]
    )
    layer_solve_results_after_source = tuple(
        [layer_solve_results_before_source[-1]]
        + [eigensolve(permittivity=p) for p in permittivities_after_source[1:]]
    )

    # Compute interior scattering matrices.
    s_matrices_interior_before_source = scattering.stack_s_matrices_interior(
        layer_solve_results_before_source, thicknesses_before_source
    )
    s_matrices_interior_after_source = scattering.stack_s_matrices_interior(
        layer_solve_results_after_source, thicknesses_after_source
    )
    s_matrix_before_source = s_matrices_interior_before_source[-1][0]
    s_matrix_after_source = s_matrices_interior_after_source[-1][0]

    # TODO: update the locations and remove the shift by `resolution`. Requires
    # updating regression test values.
    dipole_locations = [
        (pitch / 2 - resolution, pitch / 2 + offset * epi_diameter - resolution)
        for offset in dipole_y_offset
    ]
    dipoles = sources.gaussian_source(
        fwhm=jnp.asarray(dipole_fwhm),
        location=jnp.asarray(dipole_locations),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )

    zeros = jnp.zeros_like(dipoles)
    jx = jnp.concatenate([dipoles, zeros, zeros], axis=-1)
    jy = jnp.concatenate([zeros, dipoles, zeros], axis=-1)
    jz = jnp.concatenate([zeros, zeros, dipoles], axis=-1)

    (
        bwd_amplitude_ambient_end,
        fwd_amplitude_before_start,
        bwd_amplitude_before_end,
        fwd_amplitude_after_start,
        bwd_amplitude_after_end,
        fwd_amplitude_substrate_start,
    ) = sources.amplitudes_for_source(
        jx=jx,
        jy=jy,
        jz=jz,
        s_matrix_before_source=s_matrix_before_source,
        s_matrix_after_source=s_matrix_after_source,
    )

    # Compute the Poynting flux just before the source, after the source,
    # and in the ambient, to allow calculation of the extraction efficiency.
    fwd_amplitude_before_end = fields.propagate_amplitude(
        amplitude=fwd_amplitude_before_start,
        distance=s_matrix_before_source.end_layer_thickness,
        layer_solve_result=s_matrix_before_source.end_layer_solve_result,
    )
    fwd_flux_before_end, bwd_flux_before_end = fields.directional_poynting_flux(
        forward_amplitude=fwd_amplitude_before_end,
        backward_amplitude=bwd_amplitude_before_end,
        layer_solve_result=s_matrix_before_source.end_layer_solve_result,
    )
    bwd_amplitude_after_start = fields.propagate_amplitude(
        amplitude=bwd_amplitude_after_end,
        distance=s_matrix_after_source.start_layer_thickness,
        layer_solve_result=s_matrix_after_source.start_layer_solve_result,
    )
    fwd_flux_after_start, bwd_flux_after_start = fields.directional_poynting_flux(
        forward_amplitude=fwd_amplitude_after_start,
        backward_amplitude=bwd_amplitude_after_start,
        layer_solve_result=s_matrix_after_source.start_layer_solve_result,
    )
    _, bwd_flux_ambient_end = fields.directional_poynting_flux(
        forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
        backward_amplitude=bwd_amplitude_ambient_end,
        layer_solve_result=s_matrix_before_source.start_layer_solve_result,
    )

    # Compute the total forward and backward flux resulting from the source. The
    # forward flux from the source is the difference between the forward flux just
    # after the source, and the forward flux just before the source. The backward
    # flux is defined analogously.
    fwd_flux_from_source = fwd_flux_after_start - fwd_flux_before_end
    bwd_flux_from_source = bwd_flux_before_end - bwd_flux_after_start

    # Compute total power emitted by the source by summing over the Brillouin
    # zone points and Fourier orders.
    assert fwd_flux_from_source.shape == brillouin_grid_shape + (
        2 * expansion.num_terms,
        jx.shape[-1],
    )
    forward_emitted_power = jnp.sum(fwd_flux_from_source, axis=(-4, -3, -2))
    backward_emitted_power = -jnp.sum(bwd_flux_from_source, axis=(-4, -3, -2))
    total_emitted_power = forward_emitted_power + backward_emitted_power

    total_extracted_power = -jnp.sum(bwd_flux_ambient_end, axis=(-4, -3, -2))
    assert total_extracted_power.shape == (jx.shape[-1],)

    # Calculate the per-dipole extraction efficiency.
    extraction_efficiency = total_extracted_power / total_emitted_power

    # Compute the fields inside the structure.
    amplitudes_interior = fields.stack_amplitudes_interior_with_source(
        s_matrices_interior_before_source=s_matrices_interior_before_source,
        s_matrices_interior_after_source=s_matrices_interior_after_source,
        backward_amplitude_before_end=bwd_amplitude_before_end,
        forward_amplitude_after_start=fwd_amplitude_after_start,
    )
    efields, hfields, (x, y, z) = fields.stack_fields_3d_auto_grid(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=(
            layer_solve_results_before_source + layer_solve_results_after_source
        ),
        layer_thicknesses=thicknesses_before_source + thicknesses_after_source,
        grid_spacing=resolution_fields,
        num_unit_cells=brillouin_grid_shape,
    )

    return extraction_efficiency, total_emitted_power, efields, hfields, (x, y, z)


def uled_structure(
    permittivity_ambient: complex,
    permittivity_passivation: complex,
    permittivity_epi: complex,
    permittivity_metal: complex,
    thickness_ambient: float,
    thickness_epi_top: float,
    thickness_epi_bottom: float,
    thickness_passivation: float,
    thickness_metal: float,
    pitch: float,
    epi_diameter: float,
    resolution: float,
) -> Tuple[
    Tuple[jnp.ndarray, ...],
    Tuple[jnp.ndarray, ...],
    Tuple[jnp.ndarray, ...],
    Tuple[jnp.ndarray, ...],
    basis.LatticeVectors,
]:
    """Returns quantities that describe the uLED structure."""

    primitive_lattice_vectors = basis.LatticeVectors(
        u=basis.X * pitch, v=basis.Y * pitch
    )

    # Generate the permittivity for the cross section below the epi
    # structure, including only the metal and the passivation.
    outer_circle_diameter = epi_diameter + thickness_passivation * 2
    inside_outer_circle = circle_mask(
        pitch, outer_circle_diameter, resolution, x_offset=0, y_offset=0
    )
    permittivity_cross_section_passivation_only = jnp.where(
        inside_outer_circle, permittivity_passivation, permittivity_metal
    )

    # Permittivity for the cross section also including the epi.
    inside_inner_circle = circle_mask(
        pitch, epi_diameter, resolution, x_offset=0, y_offset=0
    )
    permittivity_cross_section_epi = jnp.where(
        inside_inner_circle,
        permittivity_epi,
        permittivity_cross_section_passivation_only,
    )

    # Generate the list of permittivities. For those which are scalars,
    # we add singleton spatial dimensions. Note that the epi region is
    # split into two layers, between which the source will be inserted.
    permittivities_before_source = [
        jnp.asarray(permittivity_ambient)[jnp.newaxis, jnp.newaxis],
        permittivity_cross_section_epi,
    ]
    permittivities_after_source = [
        permittivity_cross_section_epi,
        permittivity_cross_section_passivation_only,
        jnp.asarray(permittivity_metal)[jnp.newaxis, jnp.newaxis],
    ]

    thicknesses_before_source = [
        jnp.asarray(thickness_ambient),
        jnp.asarray(thickness_epi_top),
    ]
    thicknesses_after_source = [
        jnp.asarray(thickness_epi_bottom),
        jnp.asarray(thickness_passivation),
        jnp.asarray(thickness_metal),
    ]

    return (
        tuple(permittivities_before_source),
        tuple(permittivities_after_source),
        tuple(thicknesses_before_source),
        tuple(thicknesses_after_source),
        primitive_lattice_vectors,
    )


def circle_mask(
    pitch: float,
    diameter: float,
    resolution: float,
    x_offset: float,
    y_offset: float,
) -> jnp.ndarray:
    """Returns a mask that is `True` for a centered circular feature."""
    x, y = jnp.meshgrid(
        jnp.arange(-pitch / 2 + resolution / 2, pitch / 2, resolution),
        jnp.arange(-pitch / 2 + resolution / 2, pitch / 2, resolution),
        indexing="ij",
    )
    distance = jnp.sqrt((x - x_offset) ** 2 + (y - y_offset) ** 2)
    return distance < diameter / 2


if __name__ == "__main__":
    for approximate_num_terms in (200, 400, 600, 800, 1000, 1200, 1500, 2000, 2500):
        t0 = time.time()
        extraction_efficiency, total_emitted_power, *_ = simulate_uled(
            approximate_num_terms=approximate_num_terms,
        )
        print(
            f"Results with approximate_num_terms = {approximate_num_terms} (time = {time.time() - t0:.2f}s):\n"
            f"    Extraction efficiency = {jnp.squeeze(extraction_efficiency * 100)}%\n"
            f"      Total emitted power = {jnp.squeeze(total_emitted_power)}"
        )
