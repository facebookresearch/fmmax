"""A 1D anisotropic grating example.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax.numpy as jnp

from fmmax import basis, fields, fmm, scattering, utils

WAVELENGTH_NM: jnp.ndarray = jnp.array([450.0, 550.0, 620.0])
POLAR_ANGLE: jnp.ndarray = jnp.array([0.0])
AZIMUTHAL_ANGLE: jnp.ndarray = jnp.array([0.0])


def simulate_grating(
    permittivity_ambient: complex = 1.0 + 0.0j,
    wavelength_nm: jnp.ndarray = WAVELENGTH_NM,
    polar_angle: jnp.ndarray = POLAR_ANGLE,
    azimuthal_angle: jnp.ndarray = AZIMUTHAL_ANGLE,
    pitch_nm: float = 180.0,
    grating_width_nm: float = 60.0,
    grating_thickness_nm: float = 80.0,
    resolution_nm: float = 1.0,
    approximate_num_terms: int = 10,
    truncation: basis.Truncation = basis.Truncation.CIRCULAR,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the TE- and TM-polarized reflection from a 1D stripe grating.

    Args:
        permittivity_ambient: The permittivity of the ambient.
        wavelength_nm: The excitation wavelength, in nanometers.
        polar_angle: The polar angle of the incident plane wave.
        azimuthal_angle: The azimuthal angle of the incident plane wave.
        pitch_nm: The grating pitch, in nanometers.
        grating_width_nm: The width of the lines comprising the grating.
        grating_thickness_nm: The height of the grating.
        resolution_nm: The rasterization resolution for patterned layers.
        approximate_num_terms: The approximate number of terms used in the plane
            wave expansion of the fields.
        truncation: Determines the truncation of the expansion.

    Returns:
        The number of terms in the expansion, and the incident, reflected, and
        transmitted power for each order.
    """
    # Add spatial dimensions to the scalar ambient permittivity.
    permittivity_ambient_ = jnp.asarray(permittivity_ambient)[jnp.newaxis, jnp.newaxis]
    del permittivity_ambient

    # Define the permittivity tensor elements for the grating gap.
    permittivities_grating_gap = (
        permittivity_ambient_,  # xx
        jnp.zeros_like(permittivity_ambient_),  # xy
        jnp.zeros_like(permittivity_ambient_),  # yx
        permittivity_ambient_,  # yy
        permittivity_ambient_,  # zz
    )

    # Get the permittivity tensor elements for the anisotropic, and add the spatial dimensions.
    permittivities_anisotropic = permittivity_tensor_elements_anisotropic(wavelength_nm)
    permittivities_grating_tooth = [
        p[..., jnp.newaxis, jnp.newaxis] for p in permittivities_anisotropic
    ]

    # Get the permittivity tensor elements for the grating using interpolation.
    density = grating_density(pitch_nm, grating_width_nm, resolution_nm)
    permittivities_grating = [
        utils.interpolate_permittivity(
            permittivity_solid=pt, permittivity_void=pg, density=density  # type: ignore[arg-type]
        )
        for pt, pg in zip(permittivities_grating_tooth, permittivities_grating_gap)
    ]

    # Add spatial dimensions to the substrate permittivity tensor elements.
    permittivities_substrate = [
        p[..., jnp.newaxis, jnp.newaxis] for p in permittivities_anisotropic
    ]

    # Compute quantities needed for the eigensolve.
    thicknesses = [jnp.zeros(()), jnp.asarray(grating_thickness_nm), jnp.zeros(())]

    # We are modeling a 1D grating with a 2D solver. To do this efficiently, we make the
    # the 2D unit cell very small along the invariant direction, so that grating vectors
    # are well allocated.
    primitive_lattice_vectors = basis.LatticeVectors(
        u=basis.X * pitch_nm, v=basis.Y * pitch_nm / (2 * approximate_num_terms)
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=truncation,
    )
    # Assert that grating vectors all have ky == 0.
    assert max(abs(expansion.basis_coefficients[:, 1])) == 0

    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength_nm,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        permittivity=permittivity_ambient_,
    )

    # Perform the appropriate layer eigensolve for each layer in the stack. Note that the
    # ordering of tensor elements in the `permittivities_grating` and `permittivities_substrate`
    # must must match those in the arguments of `eigensolve_patterned_anisotropic_media`.
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=wavelength_nm,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity_ambient_,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        ),
        fmm.eigensolve_anisotropic_media(
            wavelength_nm,
            in_plane_wavevector,
            primitive_lattice_vectors,
            *permittivities_grating,  # type: ignore[arg-type]
            expansion=expansion,  # type: ignore[misc]
            formulation=fmm.Formulation.FFT,  # type: ignore[misc]
        ),
        fmm.eigensolve_anisotropic_media(
            wavelength_nm,
            in_plane_wavevector,
            primitive_lattice_vectors,
            *permittivities_substrate,  # type: ignore[arg-type]
            expansion=expansion,  # type: ignore[misc]
            formulation=fmm.Formulation.FFT,  # type: ignore[misc]
        ),
    ]

    # Compute the scattering matrix for our stack.
    s_matrix = scattering.stack_s_matrix(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
    )

    # Generate the wave amplitudes for the plane excitation. We excite with
    # both TE-polarized and TM-polarized light.
    n = layer_solve_results[0].expansion.num_terms
    forward_amplitude_0_start = jnp.zeros((2 * n, 2), dtype=complex)
    forward_amplitude_0_start = forward_amplitude_0_start.at[0, 0].set(1)  # te
    forward_amplitude_0_start = forward_amplitude_0_start.at[n, 1].set(1)  # tm

    backward_amplitude_0_end = s_matrix.s21 @ forward_amplitude_0_start
    _, backward_amplitude_0_start = fields.colocate_amplitudes(
        forward_amplitude_0_start,
        backward_amplitude_0_end,
        z_offset=jnp.zeros(()),
        layer_solve_result=layer_solve_results[0],
        layer_thickness=jnp.asarray(thicknesses[0]),
    )
    incident_power, reflected_power = fields.amplitude_poynting_flux(
        forward_amplitude=forward_amplitude_0_start,
        backward_amplitude=backward_amplitude_0_start,
        layer_solve_result=layer_solve_results[0],
    )

    forward_amplitude_N_start = s_matrix.s11 @ forward_amplitude_0_start
    transmitted_power, _ = fields.amplitude_poynting_flux(
        forward_amplitude=forward_amplitude_N_start,
        backward_amplitude=jnp.zeros_like(forward_amplitude_N_start),
        layer_solve_result=layer_solve_results[-1],
    )

    return expansion.num_terms, incident_power, reflected_power, transmitted_power


def permittivity_tensor_elements_anisotropic(
    wavelength_nm: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the permittivity tensor elements of an anisotropic material.

    Args:
        wavelength_nm: The wavelength for which the permittivity is sought, with
            units of nanometers.

    Returns:
        The permittivity tensor elements `(exx, exy, eyx, eyy, ezz)`.
    """
    return (
        jnp.ones_like(wavelength_nm) * 2.0**2,
        jnp.zeros_like(wavelength_nm),
        jnp.zeros_like(wavelength_nm),
        jnp.ones_like(wavelength_nm) * 2.4**2,
        jnp.ones_like(wavelength_nm) * 2.8**2,
    )


def grating_density(
    pitch: float,
    grating_width: float,
    resolution: float,
) -> jnp.ndarray:
    """Returns the density for a grating.

    The density is `1` inside the grating tooth, and `0` inside the gap.

    Args:
        pitch: The grating pitch.
        grating_width: The grating width.
        resolution: The resolution of the density array.

    Returns:
        The density array, with shape `(pitch / resolution, 1)`.
    """
    x = jnp.arange(-pitch / 2, pitch / 2, resolution)
    x = x[:, jnp.newaxis]
    return (jnp.abs(x) <= grating_width / 2).astype(float)


if __name__ == "__main__":
    print(simulate_grating())
