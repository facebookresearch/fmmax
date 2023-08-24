"""A thin film antireflection coating example.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Sequence, Tuple

import jax.numpy as jnp
import numpy as onp
import scipy.optimize as spo  # type: ignore[import]

from fmmax import basis, fmm, scattering


def compute_reflection(
    refractive_indices: Sequence[jnp.ndarray],
    thicknesses: Sequence[float],
    refractive_index_ambient: jnp.ndarray,
    refractive_index_substrate: jnp.ndarray,
    wavelength: jnp.ndarray,
    incident_angle: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the reflection for a stack of thin films.

    This function can be used to compute batches of reflection coefficients, by
    providing appropriate arrays for the refractive indices, wavelength and
    incident angle.

    Args:
        refractive_indices: The refractive indices for each layer in the stack.
        thicknesses: The thickness for each layer in the stack. Units are arbitrary,
            but should match those of `wavelength`.
        refractive_index_ambient: The refractive index of the ambient, i.e. the
            medium in which the excitation plane wave originates.
        refractive_index_substrate: The refractive index of the substrate.
        wavelength: The wavelengths of the excitation.
        incident_angle: The incident angles of the excitation.

    Returns:
        The reflection for TE- and TM-polarized excitation.
    """

    refractive_indices = [jnp.asarray(r) for r in refractive_indices]
    refractive_index_ambient = jnp.asarray(refractive_index_ambient)
    refractive_index_substrate = jnp.asarray(refractive_index_substrate)

    shape = refractive_index_ambient.shape
    if refractive_index_substrate.shape != shape or any(
        [r.shape != shape for r in refractive_indices]
    ):
        raise ValueError(
            f"All refractive indices must have the same shape, but got "
            f"{[r.shape for r in refractive_indices]}"
            f"{refractive_index_ambient.shape}, and "
            f"{refractive_index_substrate.shape}."
        )

    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength,
        polar_angle=incident_angle,
        azimuthal_angle=jnp.zeros_like(incident_angle),
        permittivity=refractive_index_ambient**2,
    )
    primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=1,
        truncation=basis.Truncation.CIRCULAR,
    )

    thicknesses_with_ambient = (
        [jnp.zeros(1)] + [jnp.asarray(t) for t in thicknesses] + [jnp.zeros(1)]
    )
    permittivities = (
        [jnp.asarray(refractive_index_ambient**2)]
        + [jnp.asarray(n**2) for n in refractive_indices]
        + [jnp.asarray(refractive_index_substrate**2)]
    )
    permittivities = [p[..., jnp.newaxis, jnp.newaxis] for p in permittivities]

    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        for p in permittivities
    ]
    s_matrix = scattering.stack_s_matrix(layer_solve_results, thicknesses_with_ambient)
    r_te = s_matrix.s21[..., 0, 0]
    r_tm = s_matrix.s21[..., expansion.num_terms, expansion.num_terms]
    return r_te, r_tm


def optimize_arc(
    refractive_indices: Tuple[float, ...] = (1.45, 2.0, 1.45, 2.0),
    min_thickness: float = 1.0,
    max_thickness: float = 150.0,
    maxiter: int = 1,
) -> None:
    """Optimizes thicknesses to minimize reflection."""

    def objective(thicknesses: jnp.ndarray) -> float:
        rte, rtm = compute_reflection(
            refractive_indices=[jnp.asarray(r) for r in refractive_indices],
            thicknesses=thicknesses.tolist(),
            refractive_index_ambient=jnp.asarray(1.0),
            refractive_index_substrate=jnp.asarray(1.45),
            wavelength=jnp.asarray([450.0, 520.0, 640.0]),
            incident_angle=jnp.zeros(()),
        )
        return float((jnp.mean(jnp.abs(rte) ** 2) + jnp.mean(jnp.abs(rtm) ** 2)) / 2)

    bounds = onp.asarray([[min_thickness, max_thickness]] * len(refractive_indices))
    result = spo.differential_evolution(
        objective, bounds=bounds, tol=1e-2, maxiter=maxiter
    )

    print(result)

    return result


if __name__ == "__main__":
    optimize_arc()
