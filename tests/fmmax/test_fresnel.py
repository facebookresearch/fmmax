"""Tests refection and transmission against the Fresnel expressions."""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fields, fmm, scattering


def fresnel_rt(n1, n2, theta_i):
    """Compute reflection and transmission by the Fresnel equations."""
    theta_t = jnp.arcsin(n1 * jnp.sin(theta_i) / n2)
    cos_theta_i = jnp.cos(theta_i)
    cos_theta_t = jnp.cos(theta_t)

    rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    ts = (2 * n1 * cos_theta_i) / (n1 * cos_theta_i + n2 * cos_theta_t)

    rp = -(n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
    tp = (2 * n1 * cos_theta_i) / (n2 * cos_theta_i + n1 * cos_theta_t)

    reflection = jnp.stack([rs, rp]).astype(complex)
    transmission = jnp.stack([ts, tp]).astype(complex)

    power_reflection = jnp.abs(reflection) ** 2
    power_transmission = (
        jnp.abs(transmission) ** 2 * n2 * cos_theta_t / (n1 * cos_theta_i)
    )
    return (reflection, transmission), (power_reflection, power_transmission)


def extract_amplitude(efield):
    """Extracts the scalar amplitude from an electric field vector."""
    max_idx = jnp.argmax(jnp.abs(efield), axis=0)
    max_element = efield[max_idx, jnp.arange(efield.shape[1])]
    phase = jnp.angle(max_element)
    amplitude = jax.vmap(jnp.linalg.norm, in_axes=1)(efield)
    amplitude = amplitude * jnp.exp(1j * phase)
    direction = efield / amplitude[jnp.newaxis, :]
    return amplitude, direction


class FresnelComparisonTest(unittest.TestCase):
    @parameterized.expand(
        [
            (1.0 + 0.0j, 1.4 + 0.0j, 0.0),
            (1.0 + 0.0j, 1.4 + 0.0j, 10.0),
            (1.0 + 0.0j, 1.4 + 0.00001j, 0.0),
            (1.0 + 0.0j, 1.4 + 0.00001j, 10.0),
            (1.4 + 0.0j, 1.0 + 0.0j, 0.0),
            (1.4 + 0.0j, 1.0 + 0.0j, 10.0),
            (1.4 + 0.00001j, 1.0 + 0.0j, 0.0),
            (1.4 + 0.00001j, 1.0 + 0.0j, 10.0),
        ]
    )
    def test_validate_fmm(self, n_ambient, n_substrate, incident_angle_deg):
        wavelength = jnp.asarray(0.537)
        incident_angle = jnp.deg2rad(incident_angle_deg)

        expansion = basis.Expansion(basis_coefficients=onp.asarray([[0, 0]]))
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)

        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=wavelength,
            polar_angle=incident_angle,
            azimuthal_angle=jnp.zeros(()),
            permittivity=jnp.asarray(n_ambient, dtype=complex) ** 2,
        )

        eigensolve_fn = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

        solve_result_ambient = eigensolve_fn(
            permittivity=jnp.asarray(n_ambient, dtype=complex) ** 2
        )
        solve_result_substrate = eigensolve_fn(
            permittivity=jnp.asarray(n_substrate, dtype=complex) ** 2
        )

        layer_solve_results = (solve_result_ambient, solve_result_substrate)
        layer_thicknesses = (jnp.ones(()), jnp.ones(()))

        # Assemble scattering matrix
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )
        s_matrix = s_matrices_interior[-1][0]

        # Compute amplitudes. We excite with y-polarized and x-polarized efields.
        fwd_amplitude_ambient_start = jnp.asarray([[1, 0], [0, 1]], dtype=complex)
        fwd_amplitude_ambient_end = fields.propagate_amplitude(
            amplitude=fwd_amplitude_ambient_start,
            distance=layer_thicknesses[0],
            layer_solve_result=solve_result_ambient,
        )
        bwd_amplitude_ambient_end = s_matrix.s21 @ fwd_amplitude_ambient_start
        fwd_amplitude_substrate_start = s_matrix.s11 @ fwd_amplitude_ambient_start
        bwd_amplitude_substrate_start = jnp.zeros_like(fwd_amplitude_ambient_start)

        # Compute power reflection and transmission coefficients
        incident_flux, reflected_flux = fields.amplitude_poynting_flux(
            forward_amplitude=fwd_amplitude_ambient_end,
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=solve_result_ambient,
        )
        transmitted_flux, _ = fields.amplitude_poynting_flux(
            forward_amplitude=fwd_amplitude_substrate_start,
            backward_amplitude=bwd_amplitude_substrate_start,
            layer_solve_result=solve_result_substrate,
        )
        power_reflection = -jnp.diag(reflected_flux) / jnp.diag(incident_flux)
        power_transmission = jnp.diag(transmitted_flux) / jnp.diag(incident_flux)

        # Compute the incident, reflected, and transmitted electric fields.
        e_incident, _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd_amplitude_ambient_end,
            backward_amplitude=jnp.zeros_like(fwd_amplitude_ambient_end),
            layer_solve_result=solve_result_ambient,
        )
        e_reflected, _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=solve_result_ambient,
        )
        e_transmitted, _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd_amplitude_substrate_start,
            backward_amplitude=jnp.zeros_like(fwd_amplitude_substrate_start),
            layer_solve_result=solve_result_substrate,
        )
        # Squeeze out the Fourier coefficient axis, since we only deal with zeroth order.
        e_incident = jnp.squeeze(jnp.asarray(e_incident), axis=1)
        e_reflected = jnp.squeeze(jnp.asarray(e_reflected), axis=1)
        e_transmitted = jnp.squeeze(jnp.asarray(e_transmitted), axis=1)
        assert e_incident.shape == (3, 2)

        e_incident_amplitude, _ = extract_amplitude(e_incident)
        e_reflected_amplitude, _ = extract_amplitude(e_reflected)
        e_transmitted_amplitude, _ = extract_amplitude(e_transmitted)

        reflection = e_reflected_amplitude / e_incident_amplitude
        transmission = e_transmitted_amplitude / e_incident_amplitude

        # Compare complex reflection and transmission coefficients, and real-valued
        # power reflection and transmission coefficients.
        (
            (expected_reflection, expected_transmission),
            (expected_power_reflection, expected_power_transmission),
        ) = fresnel_rt(n1=n_ambient, n2=n_substrate, theta_i=incident_angle)
        onp.testing.assert_allclose(
            reflection, expected_reflection, rtol=1e-5, atol=1e-7
        )
        onp.testing.assert_allclose(
            transmission, expected_transmission, rtol=1e-5, atol=1e-7
        )
        onp.testing.assert_allclose(
            power_reflection, expected_power_reflection, rtol=1e-5, atol=1e-7
        )
        onp.testing.assert_allclose(
            power_transmission, expected_power_transmission, rtol=1e-5, atol=1e-7
        )
