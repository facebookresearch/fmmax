"""An example which computes wavelength-dependent reflection of metallic pillars.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax.numpy as jnp

from fmmax import basis, fields, fmm, scattering, utils

SMatricesInterior = Tuple[
    Tuple[scattering.ScatteringMatrix, scattering.ScatteringMatrix], ...
]

WAVELENGTH_NM: jnp.ndarray = jnp.arange(440, 660, 10)


# Electronvolt-nanometer; can be divided by the wavelength to obtain photon energy.
EV_NM: float = 1239.84198  # hbar * c * 2 * pi / (1 nm * wavelength_nm) in electronvolts


def _to_ev(wavelength_nm: jnp.ndarray) -> jnp.ndarray:
    return EV_NM / wavelength_nm


# Rakic Laurence-Drude model for the permittivity of Ag.
# Rakić et al. 1998, https://doi.org/10.1364/AO.37.005271

F0: float = 0.845
OMEGA_p: float = 9.010  # eV
GAMMA_0: float = 0.048  # eV

F1: float = 0.065
GAMMA_1: float = 3.886  # eV
OMEGA_1: float = 0.816  # eV

F2: float = 0.124
GAMMA_2: float = 0.452  # eV
OMEGA_2: float = 4.481  # eV

F3: float = 0.011
GAMMA_3: float = 0.065  # eV
OMEGA_3: float = 8.185  # eV

F4: float = 0.840
GAMMA_4: float = 0.916  # eV
OMEGA_4: float = 9.083  # eV

F5: float = 5.646
GAMMA_5: float = 2.419  # eV
OMEGA_5: float = 20.29  # eV


def permittivity_ag_drude(wavelength_nm: jnp.ndarray) -> jnp.ndarray:
    """Computes the permittivity of silver for the specified wavelengths."""
    omega = _to_ev(wavelength_nm)
    permittivity = (
        (1 - F0 * OMEGA_p**2 / (omega * (omega + 1j * GAMMA_0)))
        + F1 * OMEGA_p**2 / ((OMEGA_1**2 - omega**2) - 1j * omega * GAMMA_1)
        + F2 * OMEGA_p**2 / ((OMEGA_2**2 - omega**2) - 1j * omega * GAMMA_2)
        + F3 * OMEGA_p**2 / ((OMEGA_3**2 - omega**2) - 1j * omega * GAMMA_3)
        + F4 * OMEGA_p**2 / ((OMEGA_4**2 - omega**2) - 1j * omega * GAMMA_4)
        + F5 * OMEGA_p**2 / ((OMEGA_5**2 - omega**2) - 1j * omega * GAMMA_5)
    )
    return jnp.asarray(permittivity)


def pillar_density(
    pitch_nm: float,
    pillar_diameter_nm: float,
    resolution_nm: float,
) -> jnp.ndarray:
    """Returns the density for a pillar array."""
    x_nm, y_nm = jnp.meshgrid(
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        indexing="ij",
    )
    distance_nm = jnp.sqrt(x_nm**2 + y_nm**2)
    density = (distance_nm <= pillar_diameter_nm / 2).astype(float)
    return density


def simulate_pillars(
    permittivity_ambient: complex = 1.0 + 0.0j,
    permittivity_planarization: complex = 2.25 + 0.0j,
    wavelength_nm: jnp.ndarray = WAVELENGTH_NM,
    polar_angle: float = 0.0,
    pitch_nm: float = 180.0,
    pillar_diameter_nm: float = 60.0,
    ambient_thickness_nm: float = 1000.0,
    planarization_thickness_nm: float = 20.0,
    pillar_thickness_nm: float = 80.0,
    substrate_thickness_nm: float = 100.0,
    resolution_nm: float = 1.0,
    approximate_num_terms: int = 200,
    truncation: basis.Truncation = basis.Truncation.CIRCULAR,
    formulation: fmm.Formulation = fmm.Formulation.FFT,
) -> Tuple[
    int,
    jnp.ndarray,
    jnp.ndarray,
    Tuple[
        Tuple[fmm.LayerSolveResult, ...],
        Tuple[jnp.ndarray, ...],
        SMatricesInterior,
    ],
]:
    """Computes the reflection from an array of circular silver pillars.

    Args:
        permittivity_ambient: The permittivity of the ambient.
        permittivity_planarization: The permittivity of media encapsulating pillars.
        wavelength_nm: The excitation wavelengths, in nanometers.
        polar_angle: The polar angle of the incident plane wave.
        pitch_nm: The pillar pitch, in nanometers.
        pillar_diameter_nm: The diameter of the pillars, in nanometers.
        ambient_thickness_nm: The thickness of the ambient.
        planarization_thickness_nm: The thickness of the planarization layer above
            the pillars.
        pillar_thickness_nm: The thickness of the pillar layer.
        substrate_thickness_nm: The thickness of the substrate.
        resolution_nm: The rasterization resolution for patterned layers.
        approximate_num_terms: The approximate number of terms used in the plane
            wave expansion of the fields.
        truncation: Determines the truncation of the expansion.
        formulation: Specifies the formulation to be used.

    Returns:
        The number of terms in the expansion, the te- and tm-polarized reflection
        coefficients, and intermediate results from the simulation which can be used
        to compute fields. These intermediate results are the layer solve results, the
        layer thicknesses, and the interior scattering matrices.
    """
    density = pillar_density(pitch_nm, pillar_diameter_nm, resolution_nm)
    permittivity_ag = permittivity_ag_drude(wavelength_nm)[:, jnp.newaxis, jnp.newaxis]
    permittivity_pillars = utils.interpolate_permittivity(
        permittivity_solid=permittivity_ag,
        permittivity_void=jnp.asarray(permittivity_planarization),
        density=density[jnp.newaxis, :, :],
    )

    permittivities = [
        jnp.asarray(permittivity_ambient)[jnp.newaxis, jnp.newaxis, jnp.newaxis],
        jnp.asarray(permittivity_planarization)[jnp.newaxis, jnp.newaxis, jnp.newaxis],
        permittivity_pillars,
        permittivity_ag,
    ]
    thicknesses = [
        jnp.asarray(ambient_thickness_nm),
        jnp.asarray(planarization_thickness_nm),
        jnp.asarray(pillar_thickness_nm),
        jnp.asarray(substrate_thickness_nm),
    ]

    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength_nm,
        polar_angle=jnp.asarray(polar_angle),
        azimuthal_angle=jnp.zeros(()),
        permittivity=jnp.asarray(permittivity_ambient),
    )
    primitive_lattice_vectors = basis.LatticeVectors(
        u=pitch_nm * basis.X, v=pitch_nm * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=truncation,
    )
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength_nm),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]

    s_matrices_interior = scattering.stack_s_matrices_interior(
        layer_solve_results, thicknesses
    )

    # The full stack scattering matrix is just the one "below" the final layer.
    s_matrix = s_matrices_interior[-1][0]

    # The reflection coefficient for normally-incident TE-polarized plane
    # waves is the first element of the s21 block of the scattering matrix.
    r_te = s_matrix.s21[..., 0, 0]
    r_tm = s_matrix.s21[..., expansion.num_terms, expansion.num_terms]
    return (
        expansion.num_terms,
        r_te,
        r_tm,
        (tuple(layer_solve_results), tuple(thicknesses), s_matrices_interior),
    )


def compute_fields(
    layer_solve_results: Tuple[fmm.LayerSolveResult, ...],
    thicknesses: Tuple[jnp.ndarray, ...],
    s_matrices_interior: SMatricesInterior,
    resolution_nm: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the electric and magnetic fields in the simulation volume.

    Args:
        layer_solve_results: The results of the eigensolve for each layer.
        thicknesses: The thickness of each layer.
        s_matrices_interior: The interior scattering matrices.
        resolution_nm: The resolution for the grid on which fields are computed.

    Returns:
        The electric and magnetic fields, each with shape `(3, num_wavelengths, nx,
        ny, nz)`. The leading axis is for the field direction, i.e. x, y, and z.
    """

    # Excite with a TE_polarized plane wave.
    num_terms = layer_solve_results[0].expansion.num_terms
    forward_amplitude_0_start = jnp.zeros((1, 2 * num_terms, 1), dtype=complex)
    forward_amplitude_0_start = forward_amplitude_0_start.at[:, 0, 0].set(1)
    backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)

    # Compute the wave amplitudes within each layer.
    amplitudes_interior = fields.stack_amplitudes_interior(
        s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
    )

    efields, hfields, (x, y, z) = fields.stack_fields_3d_auto_grid(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=layer_solve_results,
        layer_thicknesses=thicknesses,
        grid_spacing=resolution_nm,
        num_unit_cells=(1, 1),
    )
    return efields, hfields


if __name__ == "__main__":
    print(simulate_pillars())
