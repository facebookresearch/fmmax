"""Functions related to fields in the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp

from fmmax import basis, fft, fmm, scattering, utils


def propagate_amplitude(
    amplitude: jnp.ndarray,
    distance: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
) -> jnp.ndarray:
    """Propagates waves with the given `amplitude` by `distance`.

    The propagation is along the wave direction, i.e. when `distance` is positive,
    amplitudes for forward-propagating waves are those associated with a positive
    shift along the z-axis, while the reverse is true for backward-propagating wave
    amplitudes.

    Args:
        amplitude: The amplitudes to be propagated, with a trailing batch dimension.
        distance: The distance to be propagated.
        layer_solve_result: The result of the layer eigensolve.

    Returns:
        The propagated amplitudes.
    """
    _validate_amplitudes_shape(
        (amplitude,),
        num_terms=2 * layer_solve_result.expansion.num_terms,
    )
    q = layer_solve_result.eigenvalues
    fd = jnp.exp(1j * q[..., jnp.newaxis] * distance)
    return amplitude * fd


def colocate_amplitudes(
    forward_amplitude_start: jnp.ndarray,
    backward_amplitude_end: jnp.ndarray,
    z_offset: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the forward- and backward-propagating wave amplitudes at `z_offset`.

    The calculation is for a batch of amplitudes, with the batch dimension being
    final dimension.

    Args:
        forward_amplitude_start: The amplitude of the forward eigenmodes at the
            start of the layer, with a trailing batch dimension.
        backward_amplitude_end: The amplitude of the backward eigenmodes at the
            end of the layer.
        z_offset: The location where the colocated amplitudes are sought, as an
            offset from the start of the layer.
        layer_solve_result: The result of the layer eigensolve.
        layer_thickness: The thickness of the layer.

    Returns:
        The forward- and backward-propagating wave amplitudes at `z`.
    """
    _validate_amplitudes_shape(
        (forward_amplitude_start, backward_amplitude_end),
        num_terms=2 * layer_solve_result.expansion.num_terms,
    )
    return (
        propagate_amplitude(
            amplitude=forward_amplitude_start,
            distance=z_offset,
            layer_solve_result=layer_solve_result,
        ),
        propagate_amplitude(
            amplitude=backward_amplitude_end,
            distance=layer_thickness - z_offset,
            layer_solve_result=layer_solve_result,
        ),
    )


def _validate_amplitudes_shape(
    amplitudes: Tuple[jnp.ndarray, ...], num_terms: int
) -> None:
    """Validates the shapes for the `amplitudes`."""
    if (
        amplitudes[0].ndim < 2
        or amplitudes[0].shape[-2] != num_terms
        or not utils.batch_compatible_shapes(*[a.shape for a in amplitudes])
    ):
        raise ValueError(
            f"All amplitudes must have matching shape `(..., {num_terms}, "
            f"amplitudes_batch_size)` but got shapes {[a.shape for a in amplitudes]}."
        )


def amplitude_poynting_flux(
    forward_amplitude: jnp.ndarray,
    backward_amplitude: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns total Poynting flux for forward and backward eigenmodes.

    This function decomposes the total field into components associated with
    the forward and backward amplitudes, and returns the time-average flux in
    each order for these two components. The calculation follows section 5.1
    of [2012 Liu].

    In the general case, a forward eigenmode may actually have negative
    Poynting flux, and therefore the quantities computed by this function
    should not be interpreted as the total forward and backward flux, but only
    the total flux associated with the forward and backward eigenmodes.

    If the total forward and backward flux is desired, `directional_poynting_flux`
    should be used instead. This function should only be used in the specific
    case where the flux associated with the forward and backward eigenmodes is
    needed.

    Args:
        forward_amplitude: The amplitude of the forward eigenmodes, with a
            trailing batch dimension.
        backward_amplitude: The amplitude of the backward eigenmodes, at the
            same location in space as the `forward_amplitude`.
        layer_solve_result: The results of the layer eigensolve.

    Returns:
        The Poynting flux associated with the forward and backward eigenmodes.
    """
    _validate_amplitudes_shape(
        (forward_amplitude, backward_amplitude),
        num_terms=2 * layer_solve_result.expansion.num_terms,
    )

    alpha_h = layer_solve_result.eigenvectors @ forward_amplitude
    beta_h = layer_solve_result.eigenvectors @ backward_amplitude

    A = _poynting_flux_a_matrix(layer_solve_result)
    alpha_e = A @ forward_amplitude
    beta_e = A @ backward_amplitude

    s_forward = jnp.asarray(0.5) * (
        (jnp.conj(alpha_e) * alpha_h + jnp.conj(alpha_h) * alpha_e)
        + (jnp.conj(beta_h) * alpha_e - jnp.conj(beta_e) * alpha_h)
    )
    s_backward = jnp.asarray(0.5) * (
        -(jnp.conj(beta_e) * beta_h + jnp.conj(beta_h) * beta_e)
        + jnp.conj((jnp.conj(beta_h) * alpha_e - jnp.conj(beta_e) * alpha_h))
    )
    return jnp.real(s_forward), jnp.real(s_backward)


def directional_poynting_flux(
    forward_amplitude: jnp.ndarray,
    backward_amplitude: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns total forward and backward Poynting flux.

    This function decomposes the total field into components resulting from the
    the eigenmodes with positive and negative Poynting flux, and returns the
    time-average flux in each order for these two components. The calculation
    follows section 5.1 of [2012 Liu].

    In the general case, a forward eigenmode may actually have negative
    Poynting flux, and so e.g. it may occur that a one-hot forward amplitude
    vector yields zero forward flux and nonzero backward flux.

    If the flux associated with the forward and backward eigenmodes is desired,
    `amplitude_poynting_flux` should be used instead. This function serves the
    more typical case where the total forward flux and total backward flux is
    desired.

    Args:
        forward_amplitude: The amplitude of the forward eigenmodes, with a
            trailing batch dimension.
        backward_amplitude: The amplitude of the backward eigenmodes, at the
            same location in space as the `forward_amplitude`.
        layer_solve_result: The results of the layer eigensolve.

    Returns:
        The Poynting flux associated with the forward and backward eigenmodes.
    """
    eigenmode_flux = eigenmode_poynting_flux(layer_solve_result)
    fwd_flux_eigenmode = eigenmode_flux > 0
    fwd_flux_eigenmode = fwd_flux_eigenmode[..., jnp.newaxis]

    # The eigenmodes may not all have positive Poynting flux, i.e. there may be
    # eigenmodes which carry energy backward assoicated with `forward_amplitude`.
    # We break apart `forward_amplitude` and `backward_amplitude` to separate
    # those which carry energy forward from those which carry energy backward.
    forward_amplitude_fwd_eigenmode = jnp.where(
        fwd_flux_eigenmode, forward_amplitude, 0.0
    )
    forward_amplitude_bwd_eigenmode = jnp.where(
        fwd_flux_eigenmode, 0.0, forward_amplitude
    )
    backward_amplitude_fwd_eigenmode = jnp.where(
        ~fwd_flux_eigenmode, backward_amplitude, 0.0
    )
    backward_amplitude_bwd_eigenmode = jnp.where(
        ~fwd_flux_eigenmode, 0.0, backward_amplitude
    )

    A = _poynting_flux_a_matrix(layer_solve_result)
    phi = layer_solve_result.eigenvectors

    def _poynting_flux_subset(
        a_total: jnp.ndarray,
        b_total: jnp.ndarray,
        a_subset: jnp.ndarray,
        b_subset: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.asarray(0.5) * (
            jnp.conj(A @ (a_total - b_total)) * (phi @ (a_subset + b_subset))
            + jnp.conj(phi @ (a_total + b_total)) * (A @ (a_subset - b_subset))
        )

    s_forward = _poynting_flux_subset(
        a_total=forward_amplitude,
        b_total=backward_amplitude,
        a_subset=forward_amplitude_fwd_eigenmode,
        b_subset=backward_amplitude_fwd_eigenmode,
    )
    s_backward = _poynting_flux_subset(
        a_total=forward_amplitude,
        b_total=backward_amplitude,
        a_subset=forward_amplitude_bwd_eigenmode,
        b_subset=backward_amplitude_bwd_eigenmode,
    )
    return jnp.real(s_forward), jnp.real(s_backward)


def eigenmode_poynting_flux(
    layer_solve_result: fmm.LayerSolveResult,
) -> jnp.ndarray:
    """Returns the total Poynting flux for each eigenmode.

    The result is equivalent to summing over the orders of the flux calculated
    by `amplitude_poynting_flux`, if the calculation is done for each eigenmode
    with a one-hot forward amplitude vector.

    Args:
        layer_solve_result: The results of the layer eigensolve.

    Returns:
        The per-eigenmode Poynting flux, with the same shape as the eigenvalues.
    """
    alpha_h = layer_solve_result.eigenvectors
    alpha_e = _poynting_flux_a_matrix(layer_solve_result)
    s_eigenmode = jnp.asarray(0.5) * (
        (jnp.conj(alpha_e) * alpha_h + jnp.conj(alpha_h) * alpha_e)
    )
    flux = jnp.sum(jnp.real(s_eigenmode), axis=-2)
    assert flux.shape == layer_solve_result.eigenvalues.shape
    return flux


def _poynting_flux_a_matrix(layer_solve_result: fmm.LayerSolveResult) -> jnp.ndarray:
    """Computes the `A` matrix from section 5.1 of [2012 Liu]."""
    q = layer_solve_result.eigenvalues
    phi = layer_solve_result.eigenvectors
    omega_script_k = layer_solve_result.omega_script_k_matrix
    angular_frequency = utils.angular_frequency_for_wavelength(
        layer_solve_result.wavelength
    )[..., jnp.newaxis]
    return omega_script_k @ phi @ utils.diag(jnp.ones(()) / (angular_frequency * q))


def fields_from_wave_amplitudes(
    forward_amplitude: jnp.ndarray,
    backward_amplitude: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Computes the electric and magnetic fields inside a layer.

    The calculation is for a batch of amplitudes, with the batch axis being the
    final axis. There can also be leading batch axes. Accordingly, amplitudes
    should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
    dimension is preferred because it allows matrix-matrix multiplication instead
    of batched matrix-vector multiplication.

    Args:
        forward_amplitude: The amplitude of the forward-propagating waves.
        backward_amplitude: The amplitude of the backward-propagating waves,
            at the same location in space as the `forward_amplitude`.
        layer_solve_result: The results of the layer eigensolve.

    Returns:
        The electric and magnetic fields, `((ex, ey, ez), (hx, hy, hz))`.
    """
    _validate_amplitudes_shape(
        (forward_amplitude, backward_amplitude),
        num_terms=2 * layer_solve_result.expansion.num_terms,
    )

    # The matrix from equation 35 of [2012 Liu].
    matrix = field_conversion_matrix(layer_solve_result)

    # Obtain the transverse electric and magnetic fields.
    amplitudes = jnp.concatenate([forward_amplitude, backward_amplitude], axis=-2)

    fields = matrix @ amplitudes
    # The signs and ordering of fields is defined by equations 31 and 32 of [2012 Liu].
    negative_ey, ex, hx, hy = jnp.split(fields, 4, axis=-2)
    ey = -negative_ey

    # Compute the z-directed electric field, using equation 11 from [2012 Liu].
    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=layer_solve_result.in_plane_wavevector,
        primitive_lattice_vectors=layer_solve_result.primitive_lattice_vectors,
        expansion=layer_solve_result.expansion,
    )
    kx = transverse_wavevectors[..., 0, jnp.newaxis]
    ky = transverse_wavevectors[..., 1, jnp.newaxis]
    angular_frequency = utils.angular_frequency_for_wavelength(
        layer_solve_result.wavelength
    )
    angular_frequency = angular_frequency[..., jnp.newaxis, jnp.newaxis]

    # We use the the Fourier convolution matrix for the inverse of permittivity,
    # rather than inverting the Fourier convolution matrix of permittivity itself.
    # This improves convergence of the computed z-component of electric field.
    ez = (
        -layer_solve_result.inverse_z_permittivity_matrix
        @ (1j * kx * hy - 1j * ky * hx)
        / (1j * angular_frequency)
    )

    # Compute the z-directed magnetic field. The expression is similar to
    # equation 14 from [2012 Liu], but modified to allow for magnetic materials.
    hz = (
        layer_solve_result.inverse_z_permeability_matrix
        @ (1j * kx * ey - 1j * ky * ex)
        / (1j * angular_frequency)
    )

    assert ex.shape == ey.shape == ez.shape == hx.shape == hy.shape == hz.shape
    return (ex, ey, ez), (hx, hy, hz)


def field_conversion_matrix(layer_solve_result: fmm.LayerSolveResult) -> jnp.ndarray:
    """Returns the matrix which converts wave amplitudes to transverse fields."""
    # The matrix is from equation 35 of [2012 Liu].
    q = layer_solve_result.eigenvalues
    phi = layer_solve_result.eigenvectors
    omega_script_k = layer_solve_result.omega_script_k_matrix
    angular_frequency = utils.angular_frequency_for_wavelength(
        layer_solve_result.wavelength
    )[..., jnp.newaxis]

    # Note that there is a factor of `angular_frequency` in the denominator here, which
    # differs from equation 35 in [2012 Liu]. This is an error in that reference, and
    # the factor is actually present e.g. in equation 59.
    mat = omega_script_k @ phi @ utils.diag(jnp.ones(()) / (angular_frequency * q))
    return jnp.block([[mat, -mat], [phi, phi]])


# -----------------------------------------------------------------------------
# Functions compute fields at grid locations in a single xy plane.
# -----------------------------------------------------------------------------


# Type for functions that compute fields at grid locations.
FieldsXYSliceFn = Callable[
    [
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Ex, Ey, Ez Fourier amplitudes
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Hx, Hy, Hz Fourier amplitudes
        fmm.LayerSolveResult,
    ],
    Tuple[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Ex, Ey, Ez
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Hx, Hy, Hz
        Tuple[jnp.ndarray, jnp.ndarray],  # x, y
    ],
]


def fields_on_grid(
    electric_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    layer_solve_result: fmm.LayerSolveResult,
    shape: Tuple[int, int],
    num_unit_cells: Tuple[int, int],
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray],
]:
    """Transforms the fields from fourier representation to the grid.

    The fields within an array of unit cells is returned, with the number of
    cells in each direction given by `num_unit_cells`.

    The calculation is for a batch of fields, with the batch axis being the
    final axis. There can also be leading batch axes. Accordingly, fields
    should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
    dimension is preferred because it allows matrix-matrix multiplication instead
    of batched matrix-vector multiplication.

    Args:
        electric_field: `(ex, ey, ez)` electric field Fourier amplitudes.
        magnetic_field: `(hx, hy, hz)` magnetic field Fourier amplitudes.
        layer_solve_result: The results of the layer eigensolve.
        shape: The shape of the grid.
        num_unit_cells: The number of unit cells along each direction.

    Returns:
        The electric field `(ex, ey, ez)`, magnetic field `(hx, hy, hz)`,
        and the grid coordinates `(x, y)`.
    """
    _validate_amplitudes_shape(
        electric_field + magnetic_field,
        num_terms=layer_solve_result.expansion.num_terms,
    )
    x, y = basis.unit_cell_coordinates(
        primitive_lattice_vectors=layer_solve_result.primitive_lattice_vectors,
        shape=shape,
        num_unit_cells=num_unit_cells,
    )
    kx = layer_solve_result.in_plane_wavevector[..., 0, jnp.newaxis, jnp.newaxis]
    ky = layer_solve_result.in_plane_wavevector[..., 1, jnp.newaxis, jnp.newaxis]
    phase = jnp.exp(1j * (kx * x + ky * y))[..., jnp.newaxis]
    assert (
        x.shape[-2:]
        == y.shape[-2:]
        == (shape[0] * num_unit_cells[0], shape[1] * num_unit_cells[1])
    )

    def _field_on_grid(fourier_field):
        field = fft.ifft(fourier_field, layer_solve_result.expansion, shape, axis=-2)
        return jnp.tile(field, num_unit_cells + (1,))

    ex, ey, ez = electric_field
    grid_electric_field = (
        _field_on_grid(ex) * phase,
        _field_on_grid(ey) * phase,
        _field_on_grid(ez) * phase,
    )

    hx, hy, hz = magnetic_field
    grid_magnetic_field = (
        _field_on_grid(hx) * phase,
        _field_on_grid(hy) * phase,
        _field_on_grid(hz) * phase,
    )
    return grid_electric_field, grid_magnetic_field, (x, y)


def fields_on_coordinates(
    electric_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    layer_solve_result: fmm.LayerSolveResult,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray],
]:
    """Computes the fields at specified coordinates.

    The calculation is for a batch of fields, with the batch axis being the
    final axis. There can also be leading batch axes. Accordingly, fields
    should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
    dimension is preferred because it allows matrix-matrix multiplication instead
    of batched matrix-vector multiplication.

    Args:
        electric_field: `(ex, ey, ez)` electric field Fourier amplitudes.
        magnetic_field: `(hx, hy, hz)` magnetic field Fourier amplitudes.
        layer_solve_result: The results of the layer eigensolve.
        x: The x-coordinates where the fields are sought.
        y: The y-coordinates where the fields are sought, with shape matching
            that of `x`.

    Returns:
        The electric field `(ex, ey, ez)`, magnetic field `(hx, hy, hz)`,
        and the grid coordinates `(x, y)`. The field arrays each have shape
        `batch_shape + coordinates_shape + (num_amplitudes)`.
    """
    _validate_amplitudes_shape(
        electric_field + magnetic_field,
        num_terms=layer_solve_result.expansion.num_terms,
    )
    assert x.shape == y.shape

    coordinates_shape = x.shape
    ex_shape = electric_field[0].shape
    field_shape = ex_shape[:-2] + coordinates_shape + (ex_shape[-1],)

    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=layer_solve_result.in_plane_wavevector,
        primitive_lattice_vectors=layer_solve_result.primitive_lattice_vectors,
        expansion=layer_solve_result.expansion,
    )
    kx = transverse_wavevectors[..., 0, jnp.newaxis]
    ky = transverse_wavevectors[..., 1, jnp.newaxis]

    def _field_at_coordinates(fourier_field):
        field = (
            fourier_field[..., jnp.newaxis, :]
            * jnp.exp(1j * (kx * x.flatten() + ky * y.flatten()))[..., jnp.newaxis]
        )
        field = jnp.sum(field, axis=-3)
        return field.reshape(field_shape)

    ex, ey, ez = electric_field
    grid_electric_field = (
        _field_at_coordinates(ex),
        _field_at_coordinates(ey),
        _field_at_coordinates(ez),
    )

    hx, hy, hz = magnetic_field
    grid_magnetic_field = (
        _field_at_coordinates(hx),
        _field_at_coordinates(hy),
        _field_at_coordinates(hz),
    )
    return grid_electric_field, grid_magnetic_field, (x, y)


# -----------------------------------------------------------------------------
# Functions to wave amplitudes inside a stack of layers.
# -----------------------------------------------------------------------------


def stack_amplitudes_interior(
    s_matrices_interior: Tuple[
        Tuple[scattering.ScatteringMatrix, scattering.ScatteringMatrix], ...
    ],
    forward_amplitude_0_start: jnp.ndarray,
    backward_amplitude_N_end: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]:
    """Computes the wave amplitudes at interior layers within a stack.

    The calculation is for a batch of amplitudes, with the batch axis being the
    final axis. There can also be leading batch axes. Accordingly, amplitudes
    should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
    dimension is preferred because it allows matrix-matrix multiplication instead
    of batched matrix-vector multiplication.

    Args:
        s_matrices_interior: The scattering matrices for the substacks before
            and after each layer, as computed by `stack_s_matrices_interior`.
        forward_amplitude_0_start: The forward-propagating wave amplitude at the
            start of the first layer of the stack.
        backward_amplitude_N_end: The backward-propagating wave amplitude at the
            end of the last layer of the stack.

    Returns:
        The forward- and backward-propagating wave amplitude for each layer,
        defined at the start and end of each layer, respectively.
    """
    return tuple(
        [
            amplitudes_interior(
                s_matrix_before=s_matrix_before,
                s_matrix_after=s_matrix_after,
                forward_amplitude_0_start=forward_amplitude_0_start,
                backward_amplitude_N_end=backward_amplitude_N_end,
            )
            for (s_matrix_before, s_matrix_after) in s_matrices_interior
        ]
    )


def stack_amplitudes_interior_with_source(
    s_matrices_interior_before_source: Tuple[
        Tuple[scattering.ScatteringMatrix, scattering.ScatteringMatrix], ...
    ],
    s_matrices_interior_after_source: Tuple[
        Tuple[scattering.ScatteringMatrix, scattering.ScatteringMatrix], ...
    ],
    backward_amplitude_before_end: jnp.ndarray,
    forward_amplitude_after_start: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]:
    """Computes the wave amplitudes in the case of an internal source.

    Args:
        s_matrices_interior_before_source: The interior scattering matrices for
            the layer substrack before the source, as computed by
            `stack_s_matrices_interior`.
        s_matrices_interior_after_source: The interior scattering matrices for
            the layer substack after the source.
        backward_amplitude_before_end: The backward-going wave amplitude at the
            end of the layer before the source.
        forward_amplitude_after_start: The forward-going wave amplitude at the
            start of the layer after the source.

    Returns:
        The forward- and backward-propagating wave amplitude for each layer,
        defined at the start and end of each layer, respectively.
    """
    amplitudes_before = stack_amplitudes_interior(
        s_matrices_interior=s_matrices_interior_before_source,
        forward_amplitude_0_start=jnp.zeros_like(backward_amplitude_before_end),
        backward_amplitude_N_end=backward_amplitude_before_end,
    )
    amplitudes_after = stack_amplitudes_interior(
        s_matrices_interior=s_matrices_interior_after_source,
        forward_amplitude_0_start=forward_amplitude_after_start,
        backward_amplitude_N_end=jnp.zeros_like(forward_amplitude_after_start),
    )
    return amplitudes_before + amplitudes_after


def amplitudes_interior(
    s_matrix_before: scattering.ScatteringMatrix,
    s_matrix_after: scattering.ScatteringMatrix,
    forward_amplitude_0_start: jnp.ndarray,
    backward_amplitude_N_end: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the wave amplitudes at an interior layer within a stack.

    The calculation is for a batch of amplitudes, with the batch axis being the
    final axis. There can also be leading batch axes. Accordingly, amplitudes
    should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
    dimension is preferred because it allows matrix-matrix multiplication instead
    of batched matrix-vector multiplication.

    Args:
        s_matrix_before: The scattering matrix for the substack before the layer.
        s_matrix_after: The scattering matrix for the substack after the layer.
        forward_amplitude_0_start: The forward-propagating wave amplitude at the
            start of the first layer of the stack.
        backward_amplitude_N_end: The backward-propagating wave amplitude at the
            end of the last layer of the stack.

    Returns:
        The forward- and backward-propagating wave amplitude in the layer, defined
        at layer start and end, respectively.
    """
    _validate_amplitudes_shape(
        (forward_amplitude_0_start, backward_amplitude_N_end),
        num_terms=2 * s_matrix_before.start_layer_solve_result.expansion.num_terms,
    )
    num = forward_amplitude_0_start.shape[-2]

    # Solve for the interior forward-going wave amplitude at the start of the layer.
    forward_rhs = (
        s_matrix_before.s11 @ forward_amplitude_0_start
        + s_matrix_before.s12 @ s_matrix_after.s22 @ backward_amplitude_N_end
    )
    forward_mat = jnp.eye(num) - s_matrix_before.s12 @ s_matrix_after.s21
    forward_amplitude_i_start = jnp.linalg.solve(forward_mat, forward_rhs)

    # The interior backward-going wave amplitude can now be directly computed.
    backward_amplitude_i_end = (
        s_matrix_after.s21 @ forward_amplitude_i_start
        + s_matrix_after.s22 @ backward_amplitude_N_end
    )
    return forward_amplitude_i_start, backward_amplitude_i_end


# -----------------------------------------------------------------------------
# Functions to compute fields on the 3D real-space grid.
# -----------------------------------------------------------------------------


def stack_fields_3d_auto_grid(
    amplitudes_interior: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    grid_spacing: float,
    num_unit_cells: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a stack on the real-space grid.

    The grid is automatically determined from the layer dimensions and the resolution.

    Args:
        amplitudes_interior: The forward- and backward-propagating wave amplitude
            for each layer, defined at the start and end of each layer, respectively.
        layer_solve_results: The results of the layer eigensolve for each layer.
        layer_thicknesses: The thickness of each layer.
        grid_spacing: The approximate spacing of gridpoints on which the field is
            computed. The actual grid spacing is modified to align with the layer
            and unit cell boundaries.
        num_unit_cells: The number of unit cells along each direction.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """
    primitive_lattice_vectors = layer_solve_results[0].primitive_lattice_vectors
    grid_shape = (
        int(jnp.round(jnp.linalg.norm(primitive_lattice_vectors.u) / grid_spacing)),
        int(jnp.round(jnp.linalg.norm(primitive_lattice_vectors.v) / grid_spacing)),
    )

    layer_znum = tuple([int(jnp.round(t / grid_spacing)) for t in layer_thicknesses])

    return stack_fields_3d(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=layer_solve_results,
        layer_thicknesses=layer_thicknesses,
        layer_znum=layer_znum,
        grid_shape=grid_shape,
        num_unit_cells=num_unit_cells,
    )


def stack_fields_3d(
    amplitudes_interior: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    layer_znum: Sequence[int],
    grid_shape: Tuple[int, int],
    num_unit_cells: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a stack on the real-space grid.

    Args:
        amplitudes_interior: The forward- and backward-propagating wave amplitude
            for each layer, defined at the start and end of each layer, respectively.
        layer_solve_results: The results of the layer eigensolve for each layer.
        layer_thicknesses: The thickness of each layer.
        layer_znum: The number of gridpoints in the z-direction for each layer.
        grid_shape: The shape of the xy real-space grid.
        num_unit_cells: The number of unit cells along each direction.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """
    return _stack_fields_3d(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=layer_solve_results,
        layer_thicknesses=layer_thicknesses,
        layer_znum=layer_znum,
        fields_xy_slice_fn=functools.partial(
            fields_on_grid,
            shape=grid_shape,
            num_unit_cells=num_unit_cells,
        ),
    )


def stack_fields_3d_on_coordinates(
    amplitudes_interior: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    layer_znum: Sequence[int],
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a stack at specified coordinates.

    This function may be significantly faster than `stack_fields_3d` in cases where
    fields in the full simulation domain are not required.

    Args:
        amplitudes_interior: The forward- and backward-propagating wave amplitude
            for each layer, defined at the start and end of each layer, respectively.
        layer_solve_results: The results of the layer eigensolve for each layer.
        layer_thicknesses: The thickness of each layer.
        layer_znum: The number of gridpoints in the z-direction for each layer.
        x: The x-coordinates where the fields are sought.
        y: The y-coordinates where the fields are sought, with shape matching
            that of `x`.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """
    return _stack_fields_3d(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=layer_solve_results,
        layer_thicknesses=layer_thicknesses,
        layer_znum=layer_znum,
        fields_xy_slice_fn=functools.partial(fields_on_coordinates, x=x, y=y),
    )


def layer_fields_3d(
    forward_amplitude_start: jnp.ndarray,
    backward_amplitude_end: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
    layer_znum: int,
    grid_shape: Tuple[int, int],
    num_unit_cells: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a layer on the real-space grid.

    Args:
        forward_amplitude_start: The forward-going wave amplitudes, defined at the
            start of the layer.
        backward_amplitude_end: The backward-going wave amplitudes, defined at the
            end of the layer.
        layer_solve_result: The results of the layer eigensolve.
        layer_thickness: The layer thickness.
        layer_znum: The number of gridpoints in the z-direction for the layer.
        grid_shape: The shape of the xy real-space grid.
        num_unit_cells: The number of unit cells along each direction.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """
    return _layer_fields_3d(
        forward_amplitude_start=forward_amplitude_start,
        backward_amplitude_end=backward_amplitude_end,
        layer_solve_result=layer_solve_result,
        layer_thickness=layer_thickness,
        layer_znum=layer_znum,
        fields_xy_slice_fn=functools.partial(
            fields_on_grid,
            shape=grid_shape,
            num_unit_cells=num_unit_cells,
        ),
    )


def layer_fields_3d_on_coordinates(
    forward_amplitude_start: jnp.ndarray,
    backward_amplitude_end: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
    layer_znum: int,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a layer at specified coordinates

    This function may be significantly faster than `layer_fields_3d` in cases where
    fields in the full simulation domain are not required.

    Args:
        forward_amplitude_start: The forward-going wave amplitudes, defined at the
            start of the layer.
        backward_amplitude_end: The backward-going wave amplitudes, defined at the
            end of the layer.
        layer_solve_result: The results of the layer eigensolve.
        layer_thickness: The layer thickness.
        layer_znum: The number of gridpoints in the z-direction for the layer.
        x: The x-coordinates where the fields are sought.
        y: The y-coordinates where the fields are sought, with shape matching
            that of `x`.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """
    return _layer_fields_3d(
        forward_amplitude_start=forward_amplitude_start,
        backward_amplitude_end=backward_amplitude_end,
        layer_solve_result=layer_solve_result,
        layer_thickness=layer_thickness,
        layer_znum=layer_znum,
        fields_xy_slice_fn=functools.partial(fields_on_coordinates, x=x, y=y),
    )


def _stack_fields_3d(
    amplitudes_interior: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    layer_znum: Sequence[int],
    fields_xy_slice_fn: FieldsXYSliceFn,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a stack on the real-space grid.

    Args:
        amplitudes_interior: The forward- and backward-propagating wave amplitude
            for each layer, defined at the start and end of each layer, respectively.
        layer_solve_results: The results of the layer eigensolve for each layer.
        layer_thicknesses: The thickness of each layer.
        layer_znum: The number of gridpoints in the z-direction for each layer.
        fields_xy_slice_fn: Computes the fields for each xy slice given the field
            Fourier amplitudes and layer solve result.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """
    _validate_matching_lengths(
        amplitudes_interior, layer_solve_results, layer_thicknesses, layer_znum
    )

    z0 = jnp.zeros(())
    zs = []
    efields = []
    hfields = []
    for layer_amplitudes, layer_solve_result, layer_thickness, znum in zip(
        amplitudes_interior,
        layer_solve_results,
        layer_thicknesses,
        layer_znum,
        strict=True,
    ):
        forward_amplitude_start, backward_amplitude_end = layer_amplitudes
        assert forward_amplitude_start.shape == backward_amplitude_end.shape
        eg, hg, (x, y, z_offset) = _layer_fields_3d(
            forward_amplitude_start=forward_amplitude_start,
            backward_amplitude_end=backward_amplitude_end,
            layer_solve_result=layer_solve_result,
            layer_thickness=layer_thickness,
            layer_znum=znum,
            fields_xy_slice_fn=fields_xy_slice_fn,
        )
        efields.append(eg)
        hfields.append(hg)
        zs.append(z_offset + z0)
        z0 += layer_thickness

    merged_efields = jnp.concatenate(efields, axis=-2)
    merged_hfields = jnp.concatenate(hfields, axis=-2)
    z = jnp.concatenate(zs)
    return merged_efields, merged_hfields, (x, y, z)


def _layer_fields_3d(
    forward_amplitude_start: jnp.ndarray,
    backward_amplitude_end: jnp.ndarray,
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
    layer_znum: int,
    fields_xy_slice_fn: FieldsXYSliceFn,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the three-dimensional fields in a layer on the real-space grid.

    Args:
        forward_amplitude_start: The forward-going wave amplitudes, defined at the
            start of the layer.
        backward_amplitude_end: The backward-going wave amplitudes, defined at the
            end of the layer.
        layer_solve_result: The results of the layer eigensolve.
        layer_thickness: The layer thickness.
        layer_znum: The number of gridpoints in the z-direction for the layer.
        fields_xy_slice_fn: Computes the fields for each xy slice given the field
            Fourier amplitudes and layer solve result.

    Returns:
        The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
    """

    assert forward_amplitude_start.shape == backward_amplitude_end.shape

    z_offset = jnp.linspace(0, layer_thickness, layer_znum)

    # Add a trailing batch dimension to `z_offset`, which matches the trailing
    # batch dimension for the amplitudes.
    amplitude_batch_size = forward_amplitude_start.shape[-1]
    z_offset_broadcast = jnp.broadcast_to(
        z_offset[:, jnp.newaxis], (layer_znum, amplitude_batch_size)
    )

    # Add a batch dimension to the amplitudes, for the offset.
    forward_amplitude_start = jnp.broadcast_to(
        forward_amplitude_start[..., jnp.newaxis, :],
        forward_amplitude_start.shape[:-1] + (layer_znum, amplitude_batch_size),
    )
    backward_amplitude_end = jnp.broadcast_to(
        backward_amplitude_end[..., jnp.newaxis, :],
        backward_amplitude_end.shape[:-1] + (layer_znum, amplitude_batch_size),
    )

    # Flatten the trailing batch dimensions for offset and amplitudes.
    z_offset_broadcast = z_offset_broadcast.flatten()
    forward_amplitude_start = forward_amplitude_start.reshape(
        forward_amplitude_start.shape[:-2] + (-1,)
    )
    backward_amplitude_end = backward_amplitude_end.reshape(
        backward_amplitude_end.shape[:-2] + (-1,)
    )

    # Compute the amplitudes everywhere in the layer. This creates a batch of
    # amplitudes. Since each of those are associated with the same layer solve
    # result, we ensure that the batch axis is the final axis, since this is
    # the "fast" batch axis for field computations.
    forward_amplitude, backward_amplitude = colocate_amplitudes(
        forward_amplitude_start,
        backward_amplitude_end,
        z_offset=z_offset_broadcast,
        layer_solve_result=layer_solve_result,
        layer_thickness=layer_thickness,
    )
    assert forward_amplitude.shape[-1] == z_offset_broadcast.size

    ef, hf = fields_from_wave_amplitudes(
        forward_amplitude=forward_amplitude,
        backward_amplitude=backward_amplitude,
        layer_solve_result=layer_solve_result,
    )
    eg_tuple, hg_tuple, (x, y) = fields_xy_slice_fn(
        electric_field=ef,  # type: ignore[call-arg]
        magnetic_field=hf,  # type: ignore[call-arg]
        layer_solve_result=layer_solve_result,  # type: ignore[call-arg]
    )
    eg = jnp.asarray(eg_tuple)
    hg = jnp.asarray(hg_tuple)

    # Restore the original amplitude batch dimension.
    eg = jnp.reshape(eg, eg.shape[:-1] + (layer_znum, amplitude_batch_size))
    hg = jnp.reshape(hg, hg.shape[:-1] + (layer_znum, amplitude_batch_size))

    return eg, hg, (x, y, z_offset)


def _validate_matching_lengths(*sequences: Sequence) -> None:
    """Validates that all of `args` have matching length."""
    lengths = [len(s) for s in sequences]
    if not all([l == lengths[0] for l in lengths]):
        raise ValueError(f"Encountered incompatible lengths, got lengths of {lengths}")
