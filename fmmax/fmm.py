"""Functions related to layer eigenmode calculation for the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import enum
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from fmmax import basis, fft, utils, vector

# xx, xy, yx, yy, and zz components of permittivity or permeability.
_TensorComponents = Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]


@enum.unique
class Formulation(enum.Enum):
    """Enumerates supported Fourier modal method formulations."""

    FFT: str = "fft"
    JONES_DIRECT: str = vector.JONES_DIRECT
    JONES: str = vector.JONES
    NORMAL: str = vector.NORMAL
    POL: str = vector.POL


def eigensolve_isotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> "LayerSolveResult":
    """Performs the eigensolve for a layer with isotropic permittivity.

    This function performs either a uniform-layer or patterned-layer eigensolve,
    depending on the shape of the trailing dimensions of a given layer permittivity.
    When the final two dimensions have shape `(1, 1)`, the layer is treated as
    uniform. Otherwise, it is patterned.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        The `LayerSolveResult`.
    """
    if permittivity.shape[-2:] == (1, 1):
        _eigensolve_fn = _eigensolve_uniform_isotropic_media
    else:
        _eigensolve_fn = functools.partial(
            _eigensolve_patterned_isotropic_media, formulation=formulation
        )

    return _eigensolve_fn(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=permittivity,
        expansion=expansion,
    )


def eigensolve_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    permittivity_zz: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> "LayerSolveResult":
    """Performs the eigensolve for a layer with anisotropic permittivity.

    This function performs either a uniform-layer or patterned-layer eigensolve,
    depending on the shape of the trailing dimensions of a given layer permittivity.
    When the final two dimensions have shape `(1, 1)`, the layer is treated as
    uniform. Otherwise, it is patterned.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape `(..., nx, ny)`.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        permittivity_zz: The zz-component of the permittivity tensor.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        The `LayerSolveResult`.
    """
    return eigensolve_general_anisotropic_media(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity_xx=permittivity_xx,
        permittivity_xy=permittivity_xy,
        permittivity_yx=permittivity_yx,
        permittivity_yy=permittivity_yy,
        permittivity_zz=permittivity_zz,
        permeability_xx=jnp.ones_like(permittivity_xx),
        permeability_xy=jnp.zeros_like(permittivity_xx),
        permeability_yx=jnp.zeros_like(permittivity_xx),
        permeability_yy=jnp.ones_like(permittivity_xx),
        permeability_zz=jnp.ones_like(permittivity_xx),
        expansion=expansion,
        formulation=formulation,
        vector_field_source=None,
    )


def eigensolve_general_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    permittivity_zz: jnp.ndarray,
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    permeability_zz: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
    vector_field_source: Optional[jnp.ndarray] = None,
) -> "LayerSolveResult":
    """Performs the eigensolve for a general anistropic layer.

    Here, "general" refers to the fact that the layer material can be magnetic, i.e.
    the permeability and permittivity can be specified.

    This function performs either a uniform-layer or patterned-layer eigensolve,
    depending on the shape of the trailing dimensions of a given layer permittivity.
    When the final two dimensions have shape `(1, 1)`, the layer is treated as
    uniform. Otherwise, it is patterned.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape `(..., nx, ny)`.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        permittivity_zz: The zz-component of the permittivity tensor.
        permeability_xx: The xx-component of the permeability tensor.
        permeability_xy: The xy-component of the permeability tensor.
        permeability_yx: The yx-component of the permeability tensor.
        permeability_yy: The yy-component of the permeability tensor.
        permeability_zz: The zz-component of the permeability tensor.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.
        vector_field_source: Optional array used to calculate the vector field for
            vector formulations of the FMM. If not specified, `(permittivity_xx +
            permittivity_yy) / 2` is used. Ignored for the `FFT` formulation. Should
            have shape matching the permittivities and permeabilities.

    Returns:
        The `LayerSolveResult`.
    """
    if permittivity_xx.shape[-2:] == (1, 1):
        _eigensolve_fn = _eigensolve_uniform_general_anisotropic_media
    else:
        if vector_field_source is None:
            vector_field_source = (permittivity_xx + permittivity_yy) / 2
        _eigensolve_fn = functools.partial(
            _eigensolve_patterned_general_anisotropic_media,
            formulation=formulation,
            vector_field_source=vector_field_source,
        )

    return _eigensolve_fn(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivities=(
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
        ),
        permeabilities=(
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
        ),
        expansion=expansion,
    )


@dataclasses.dataclass
class LayerSolveResult:
    """Stores the result of a layer eigensolve.

    This eigenvalue problem is specified in equation 28 of [2012 Liu].

    Attributes:
        wavelength: The wavelength for the solve.
        in_plane_wavevector: The in-plane wavevector for the solve.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        expansion: The expansion used for the eigensolve.
        eigenvalues: The layer eigenvalues.
        eigenvectors: The layer eigenvectors.
        z_permittivity_matrix: The fourier-transformed zz-component of permittivity.
        inverse_z_permittivity_matrix: The fourier-transformed inverse of zz-component
            of permittivity.
        z_permeability_matrix: The fourier-transformed zz-component of permeability.
        inverse_z_permeability_matrix: The fourier-transformed inverse of zz-component
            of permeability.
        omega_script_k_matrix: The omega-script-k matrix from equation 26 of
            [2012 Liu], which is needed to generate the layer scattering matrix.
    """

    wavelength: jnp.ndarray
    in_plane_wavevector: jnp.ndarray
    primitive_lattice_vectors: basis.LatticeVectors
    expansion: basis.Expansion
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    z_permittivity_matrix: jnp.ndarray
    inverse_z_permittivity_matrix: jnp.ndarray
    z_permeability_matrix: jnp.ndarray
    inverse_z_permeability_matrix: jnp.ndarray
    omega_script_k_matrix: jnp.ndarray

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self.eigenvectors.shape[:-2]

    def __post_init__(self) -> None:
        """Validates shapes of the `LayerSolveResult` attributes."""

        def _incompatible(arr: jnp.ndarray, reference_shape: Tuple[int, ...]) -> bool:
            ndim_mismatch = arr.ndim != len(reference_shape)
            batch_compatible = utils.batch_compatible_shapes(arr.shape, reference_shape)
            return ndim_mismatch or not batch_compatible

        if _incompatible(self.wavelength, self.batch_shape):
            raise ValueError(
                f"`wavelength` must have compatible batch shape, but got shape {self.wavelength.shape} "
                f"when `eigenvectors` shape is {self.eigenvectors.shape}."
            )
        if _incompatible(self.in_plane_wavevector, self.batch_shape + (2,)):
            raise ValueError(
                f"`in_plane_wavevector` must have compatible batch shape, but got shape "
                f"{self.in_plane_wavevector.shape} when `eigenvectors` shape is {self.eigenvectors.shape}."
            )
        if self.expansion.num_terms * 2 != self.eigenvectors.shape[-1]:
            raise ValueError(
                f"`eigenvectors` must have shape compatible with `expansion.num_terms`, but got shape "
                f"{self.eigenvectors.shape} when `num_terms` shape is {self.expansion.num_terms}."
            )
        if self.eigenvalues.shape != self.eigenvectors.shape[:-1]:
            raise ValueError(
                f"`eigenvalues` must have compatible shape, but got shape {self.eigenvalues.shape} "
                f"when `eigenvectors` shape is {self.eigenvectors.shape}."
            )

        expected_matrix_shape = self.batch_shape + (self.expansion.num_terms,) * 2
        if _incompatible(self.inverse_z_permittivity_matrix, expected_matrix_shape):
            raise ValueError(
                f"`inverse_z_permittivity_matrix` must have shape compatible with `eigenvectors`, "
                f"but got shapes {self.inverse_z_permittivity_matrix.shape}  and {self.eigenvectors.shape}."
            )
        if _incompatible(self.z_permittivity_matrix, expected_matrix_shape):
            raise ValueError(
                f"`z_permittivity_matrix` must have shape compatible with `eigenvectors`, but got "
                f"shapes {self.z_permittivity_matrix.shape}  and {self.eigenvectors.shape}."
            )
        if _incompatible(self.inverse_z_permeability_matrix, expected_matrix_shape):
            raise ValueError(
                f"`inverse_z_permeability_matrix` must have shape compatible with `eigenvectors`, "
                f"but got shapes {self.inverse_z_permeability_matrix.shape}  and {self.eigenvectors.shape}."
            )
        if _incompatible(self.z_permeability_matrix, expected_matrix_shape):
            raise ValueError(
                f"`z_permeability_matrix` must have shape compatible with `eigenvectors`, but got "
                f"shapes {self.z_permeability_matrix.shape}  and {self.eigenvectors.shape}."
            )
        if self.omega_script_k_matrix.shape != self.eigenvectors.shape:
            raise ValueError(
                f"`omega_script_k_matrix` must have shape matching `eigenvectors`, but got "
                f"shapes {self.omega_script_k_matrix.shape}  and {self.eigenvectors.shape}."
            )


# -----------------------------------------------------------------------------
# Eigensolves for specific cases, e.g. uniform isotropic, anisotropic, etc.
# -----------------------------------------------------------------------------


def _eigensolve_uniform_isotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
) -> LayerSolveResult:
    r"""Returns the the results of a uniform isotropic layer eigensolve.

    The layer is uniform and isotropic, in the sense that the permittivity does not
    vary spatially and has no orientation dependence. In this case, the eigenvalues
    and eigenvectors can be calculated analytically.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The scalar permittivity for the layer, with shape `(..., 1, 1)`.
        expansion: The field expansion to be used.

    Returns:
        The `LayerSolveResult`.
    """
    wavelength, in_plane_wavevector, permittivity = _validate_and_broadcast(
        wavelength, in_plane_wavevector, permittivity
    )
    if permittivity.shape[-2:] != (1, 1):
        raise ValueError(
            f"Trailing axes of `permittivity` must have shape (1, 1) but got a shape "
            f"of {permittivity.shape}."
        )

    batch_shape = jnp.broadcast_shapes(
        wavelength.shape, in_plane_wavevector.shape[:-1], permittivity.shape[:-2]
    )

    num_eigenvalues = 2 * expansion.num_terms
    permittivity = jnp.squeeze(permittivity, axis=(-2, -1))

    # Transverse wavevectors are the `k + G` from equation 5 of [2012 Liu].
    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )

    # In uniform media, the eigenvectors are just the plane waves.
    eigenvectors = jnp.broadcast_to(
        jnp.eye(num_eigenvalues, dtype=complex),
        shape=batch_shape + (num_eigenvalues, num_eigenvalues),
    )

    angular_frequency = utils.angular_frequency_for_wavelength(wavelength)
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    eigenvalues = jnp.sqrt(
        (
            permittivity[..., jnp.newaxis] * angular_frequency[..., jnp.newaxis] ** 2
            - kx**2
            - ky**2
        ).astype(complex)
    )
    eigenvalues = _select_eigenvalues_sign(eigenvalues)
    eigenvalues = jnp.tile(eigenvalues, 2)

    batch_shape = eigenvalues.shape[:-1]
    inverse_z_permittivity_matrix = jnp.broadcast_to(
        1 / permittivity[..., jnp.newaxis], batch_shape + (expansion.num_terms,)
    )
    inverse_z_permittivity_matrix = utils.diag(inverse_z_permittivity_matrix)
    z_permittivity_matrix = jnp.broadcast_to(
        permittivity[..., jnp.newaxis], batch_shape + (expansion.num_terms,)
    )
    z_permittivity_matrix = utils.diag(z_permittivity_matrix)
    z_permeability_matrix = utils.diag(jnp.ones(z_permittivity_matrix.shape[:-1]))

    # The matrix from equation 26 of [2012 Liu].
    angular_frequency_squared = angular_frequency[..., jnp.newaxis, jnp.newaxis] ** 2
    angular_frequency_squared *= jnp.eye(num_eigenvalues)
    omega_script_k_matrix = angular_frequency_squared - _script_k_matrix_uniform(
        permittivity, transverse_wavevectors
    )
    return LayerSolveResult(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=z_permeability_matrix,
        omega_script_k_matrix=omega_script_k_matrix,
    )


def _eigensolve_patterned_isotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> LayerSolveResult:
    r"""Returns the results of a patterned isotropic layer eigensolve.

    The layer is patterned and isotropic, in the sense that the permittivity varies
    spatially and has no orientation dependence.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        The `LayerSolveResult`.
    """
    wavelength, in_plane_wavevector, permittivity = _validate_and_broadcast(
        wavelength, in_plane_wavevector, permittivity
    )
    (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
    ) = fourier_matrices_patterned_isotropic_media(
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=permittivity,
        expansion=expansion,
        formulation=formulation,
    )

    # Create permeability matrices for nonmagnetic materials.
    ones = jnp.ones(z_permittivity_matrix.shape[:-1])
    zeros = jnp.zeros_like(ones)
    z_permeability_matrix = utils.diag(ones)
    inverse_z_permeability_matrix = utils.diag(ones)
    transverse_permeability_matrix = jnp.block(
        [
            [utils.diag(ones), utils.diag(zeros)],
            [utils.diag(zeros), utils.diag(ones)],
        ]
    )

    return _numerical_eigensolve(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        transverse_permittivity_matrix=transverse_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        expansion=expansion,
    )


def _eigensolve_uniform_general_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: _TensorComponents,
    permeabilities: _TensorComponents,
    expansion: basis.Expansion,
) -> LayerSolveResult:
    """Returns the results of a uniform anisotropic layer eigensolve.

    The layer is uniform and anisotropic, in the sense that the permittivity does not
    vary spatially and has orientation dependence.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: `(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)`, each having shape `(..., nx, ny)`.
        permeabilities: The elements of the permeability tensor: `(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)`, each having shape `(..., nx, ny)`.
        expansion: The field expansion to be used.

    Returns:
        The `LayerSolveResult`.
    """
    if not all([p.shape[-2:] == (1, 1) for p in permittivities + permeabilities]):
        raise ValueError(
            f"Trailing axes of arrays in `permittivities` and `permeabilities` must have shape "
            f"(1, 1) but got a shapes {[p.shape for p in permittivities]} and "
            f"{[p.shape for p in permeabilities]}."
        )
    (
        wavelength,
        in_plane_wavevector,
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
        permeability_xx,
        permeability_xy,
        permeability_yx,
        permeability_yy,
        permeability_zz,
    ) = _validate_and_broadcast(
        wavelength,
        in_plane_wavevector,
        *permittivities,
        *permeabilities,
    )

    batch_shape = jnp.broadcast_shapes(
        wavelength.shape, in_plane_wavevector.shape[:-1], permittivity_xx.shape[:-2]
    )
    shape = batch_shape + (expansion.num_terms,)

    permittivity_xx = jnp.broadcast_to(jnp.squeeze(permittivity_xx, axis=-1), shape)
    permittivity_xy = jnp.broadcast_to(jnp.squeeze(permittivity_xy, axis=-1), shape)
    permittivity_yx = jnp.broadcast_to(jnp.squeeze(permittivity_yx, axis=-1), shape)
    permittivity_yy = jnp.broadcast_to(jnp.squeeze(permittivity_yy, axis=-1), shape)
    permittivity_zz = jnp.broadcast_to(jnp.squeeze(permittivity_zz, axis=-1), shape)
    z_permittivity_matrix = utils.diag(permittivity_zz)
    inverse_z_permittivity_matrix = utils.diag(1 / permittivity_zz)
    # Note that the matrix element ordering and signs differ from [2012 Liu] equation 37,
    # but are consistent with the definition in equation 15. Equation 37 is likely in error.
    transverse_permittivity_matrix = jnp.block(
        [
            [utils.diag(permittivity_yy), utils.diag(-permittivity_yx)],
            [utils.diag(-permittivity_xy), utils.diag(permittivity_xx)],
        ]
    )

    permeability_xx = jnp.broadcast_to(jnp.squeeze(permeability_xx, axis=-1), shape)
    permeability_xy = jnp.broadcast_to(jnp.squeeze(permeability_xy, axis=-1), shape)
    permeability_yx = jnp.broadcast_to(jnp.squeeze(permeability_yx, axis=-1), shape)
    permeability_yy = jnp.broadcast_to(jnp.squeeze(permeability_yy, axis=-1), shape)
    permeability_zz = jnp.broadcast_to(jnp.squeeze(permeability_zz, axis=-1), shape)
    z_permeability_matrix = utils.diag(permeability_zz)
    inverse_z_permeability_matrix = utils.diag(1 / permeability_zz)
    # Note that the matrix element ordering for the transverse permittivity and
    # permeability matrices differs.
    transverse_permeability_matrix = jnp.block(
        [
            [utils.diag(permeability_xx), utils.diag(permeability_xy)],
            [utils.diag(permeability_yx), utils.diag(permeability_yy)],
        ]
    )

    return _numerical_eigensolve(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        transverse_permittivity_matrix=transverse_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        expansion=expansion,
    )


def _eigensolve_patterned_general_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: _TensorComponents,
    permeabilities: _TensorComponents,
    expansion: basis.Expansion,
    formulation: Formulation,
    vector_field_source: jnp.ndarray,
) -> LayerSolveResult:
    """Returns the results of a patterned anisotropic layer eigensolve.

    The layer is patterned and anisotropic, in the sense that the permittivity varies
    spatially and has orientation dependence.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: `(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)`, each having shape `(..., nx, ny)`.
        permeabilities: The elements of the permeability tensor: `(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)`, each having shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.
        vector_field_source: Array used to calculate the vector field, with shape
            matching the permittivities and permeabilities.

    Returns:
        The `LayerSolveResult`.
    """
    (
        wavelength,
        in_plane_wavevector,
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
        permeability_xx,
        permeability_xy,
        permeability_yx,
        permeability_yy,
        permeability_zz,
    ) = _validate_and_broadcast(
        wavelength,
        in_plane_wavevector,
        *permittivities,
        *permeabilities,
    )
    (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        inverse_z_permeability_matrix,
        z_permeability_matrix,
        transverse_permeability_matrix,
    ) = fourier_matrices_patterned_anisotropic_media(
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivities=(
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
        ),
        permeabilities=(
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
        ),
        expansion=expansion,
        formulation=formulation,
        vector_field_source=vector_field_source,
    )
    return _numerical_eigensolve(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        transverse_permittivity_matrix=transverse_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        expansion=expansion,
    )


# -----------------------------------------------------------------------------
# Helper function used by all eigensolves done numerically.
# -----------------------------------------------------------------------------


def _numerical_eigensolve(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    z_permittivity_matrix: jnp.ndarray,
    inverse_z_permittivity_matrix: jnp.ndarray,
    transverse_permittivity_matrix: jnp.ndarray,
    z_permeability_matrix: jnp.ndarray,
    inverse_z_permeability_matrix: jnp.ndarray,
    transverse_permeability_matrix: jnp.ndarray,
    expansion: basis.Expansion,
) -> LayerSolveResult:
    r"""Returns the results of a patterned layer eigensolve.

    The layer may be anisotropic and magnetic, as determined by the provided transverse
    permittivity and permeability matrices.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: `(kx0, ky0)`.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        inverse_z_permittivity_matrix: The fourier-transformed inverse of zz-component
            of permittivity.
        z_permittivity_matrix: The fourier-transformed zz-component of permittivity.
        transverse_permittivity_matrix: The fourier-transformed transverse permittivity
            matrix from equation 15 of [2012 Liu].
        inverse_z_permeability_matrix: The fourier-transformed inverse of zz-component
            of permeability.
        z_permeability_matrix: The fourier-transformed zz-component of permeability.
        transverse_permeability_matrix: The fourier-transformed transverse permeability
            matrix.
        expansion: The field expansion to be used.

    Returns:
        The `LayerSolveResult`.
    """
    # Transverse wavevectors are the `k + G` from equation 5 of [2012 Liu].
    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )

    angular_frequency = utils.angular_frequency_for_wavelength(wavelength)

    # The k matrix from equation 23 of [2012 Liu], modified for magnetic materials.
    k_matrix = _k_matrix_patterned(z_permeability_matrix, transverse_wavevectors)

    # The script-k matrix from equation 19 of [2012 Liu].
    script_k_matrix = _script_k_matrix_patterned(
        z_permittivity_matrix, transverse_wavevectors
    )

    # The matrix from equation 26 of [2012 Liu], modified for magnetic materials.
    angular_frequency_squared = angular_frequency[..., jnp.newaxis, jnp.newaxis] ** 2
    omega_script_k_matrix = (
        angular_frequency_squared * transverse_permeability_matrix - script_k_matrix
    )

    # The matrix from equation 28 of [2012 Liu], modified for magnetic materials.
    matrix = (
        transverse_permittivity_matrix @ omega_script_k_matrix
        - k_matrix @ transverse_permeability_matrix
    )
    eigenvalues_squared, eigenvectors = utils.eig(matrix)
    eigenvalues = jnp.sqrt(eigenvalues_squared)
    eigenvalues = _select_eigenvalues_sign(eigenvalues)
    return LayerSolveResult(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        omega_script_k_matrix=omega_script_k_matrix,
    )


# -----------------------------------------------------------------------------
# Functions to compute the k- and script-k matrices.
# -----------------------------------------------------------------------------


def _script_k_matrix_uniform(
    permittivity: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the uniform-layer script-k matrix from eq. 19 of [2012 Liu]."""
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    return jnp.block(
        [
            [
                utils.diag(ky / permittivity[..., jnp.newaxis] * ky),
                utils.diag(-ky / permittivity[..., jnp.newaxis] * kx),
            ],
            [
                utils.diag(-kx / permittivity[..., jnp.newaxis] * ky),
                utils.diag(kx / permittivity[..., jnp.newaxis] * kx),
            ],
        ]
    )


def _script_k_matrix_patterned(
    z_permittivity_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the patterned-layer script-k matrix from eq. 19 of [2012 Liu]."""
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    z_inv_kx = jnp.linalg.solve(z_permittivity_matrix, utils.diag(kx))
    z_inv_ky = jnp.linalg.solve(z_permittivity_matrix, utils.diag(ky))
    return jnp.block(
        [
            [ky[..., :, jnp.newaxis] * z_inv_ky, -ky[..., :, jnp.newaxis] * z_inv_kx],
            [-kx[..., :, jnp.newaxis] * z_inv_ky, kx[..., :, jnp.newaxis] * z_inv_kx],
        ]
    )


def _k_matrix_patterned(
    z_permeability_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the k-matrix for patterned magnetic materials."""
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    z_inv_kx = jnp.linalg.solve(z_permeability_matrix, utils.diag(kx))
    z_inv_ky = jnp.linalg.solve(z_permeability_matrix, utils.diag(ky))
    return jnp.block(
        [
            [kx[..., :, jnp.newaxis] * z_inv_kx, kx[..., :, jnp.newaxis] * z_inv_ky],
            [ky[..., :, jnp.newaxis] * z_inv_kx, ky[..., :, jnp.newaxis] * z_inv_ky],
        ]
    )


# -----------------------------------------------------------------------------
# Functions for computing Fourier convolution matrices for isotropic media.
# -----------------------------------------------------------------------------


def fourier_matrices_patterned_isotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Fourier convolution matrices for patterned nonmagnetic isotropic media.

    All matrices are forms of the Fourier convolution matrices defined in equation
    8 of [2012 Liu]. For vector formulations, the transverse permittivity matrix is
    of the form E2 given in equation 51 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, e.g. a vector formulation
            or the non-vector `FFT` formulation.

    Returns:
        inverse_z_permittivity_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix.
    """
    if formulation == Formulation.FFT:
        transverse_permittivity_matrix = _transverse_permittivity_fft(
            permittivity=permittivity,
            expansion=expansion,
        )
    else:
        transverse_permittivity_matrix = _transverse_permittivity_vector(
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=formulation,
        )

    _transform = functools.partial(
        fft.fourier_convolution_matrix,
        expansion=expansion,
    )

    inverse_z_permittivity_matrix = _transform(1 / permittivity)
    z_permittivity_matrix = _transform(permittivity)
    return (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
    )


def _transverse_permittivity_fft(
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Computes the `eps` matrix from [2012 Liu] equation 15 using `fft` scheme.

    Args:
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.

    Returns:
        The transverse permittivity matrix.
    """
    eps_hat = fft.fourier_convolution_matrix(permittivity, expansion)
    zeros = jnp.zeros_like(eps_hat)
    return jnp.block([[eps_hat, zeros], [zeros, eps_hat]])


def _transverse_permittivity_vector(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation,
) -> jnp.ndarray:
    """Computes transverse permittivity matrix using a vector field methods.

    The transverse permittivity matrix is E2 given in equation 51 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.

    Returns:
        The `eps` matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    eps_hat = _transform(permittivity)
    zeros = jnp.zeros_like(eps_hat)
    eps_matrix = jnp.block([[eps_hat, zeros], [zeros, eps_hat]])

    eta = 1 / permittivity
    eta_hat = _transform(eta)
    delta_hat = eps_hat - jnp.linalg.inv(eta_hat)
    delta_matrix = jnp.block([[delta_hat, zeros], [zeros, delta_hat]])

    vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
    tx, ty = vector_fn(permittivity, primitive_lattice_vectors)

    Pxx, Pxy, Pyx, Pyy = tangent_terms(tx, ty)
    p_matrix = jnp.block(
        [[_transform(Pyy), _transform(Pyx)], [_transform(Pxy), _transform(Pxx)]]
    )
    return eps_matrix - delta_matrix @ p_matrix


def tangent_terms(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the `(Pxx, Pxy, Pyx, Pyy)` from [2012 Liu] equation 50.

    Args:
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.

    Returns:
        The terms `(Pxx, Pxy, Pyx, Pyy)`.
    """
    denom = jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
    denom_safe = jnp.where(jnp.isclose(denom, 0), 1.0, denom)

    # Compute terms from equation 50 of [2012 Liu]. Note that there is an error
    # in equation 48-51 of [2012 Liu], where x- and y- subscripts are swapped for
    # the diagonal blocks. This can be shown by deriving equation 48 from 46, and
    # can also easily be seen by considering the case of a 1D gratng periodic in
    # the x-direction. In this case, the tangent vector field has a y-component
    # only (i.e. tx = 0), and we want to recover the transverse permittivity
    # given by equation 43. This is possible only by swapping the x- and y-
    # subscripts in the diagonal terms of equations 48-51.
    #
    # Note that we also investigated both orderings numerically with the 1D
    # grating example, and found improved convergence with this correction.
    Pyy = jnp.abs(tx) ** 2 / denom_safe
    Pyx = jnp.conj(tx) * ty / denom_safe
    Pxy = tx * jnp.conj(ty) / denom_safe
    Pxx = jnp.abs(ty) ** 2 / denom_safe
    return Pxx, Pxy, Pyx, Pyy


# -----------------------------------------------------------------------------
# Functions for computing Fourier convolution matrices for anisotropic media.
# -----------------------------------------------------------------------------


def fourier_matrices_patterned_anisotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: _TensorComponents,
    permeabilities: _TensorComponents,
    expansion: basis.Expansion,
    formulation: Formulation,
    vector_field_source: jnp.ndarray,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Return Fourier convolution matrices for patterned anisotropic media.

    The transverse permittivity matrix E is defined as,

        [-Dy, Dx]^T = E [-Ey, Ex]^T

    while the transverse permeability matrix M is defined as,

        [Bx, By]^T = M [Hx, Hy]^T

    The Fourier factorization is done as for E1 given in equation 47 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: `(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)`, each having shape `(..., nx, ny)`.
        permeabilities: The elements of the permeability tensor: `(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)`, each having shape `(..., nx, ny)`.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.
        vector_field_source: Array used to calculate the vector field, with shape
            matching the permittivities and permeabilities.

    Returns:
        inverse_z_permittivity_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix from
            equation 15 of [2012 Liu], computed in the manner prescribed by
            `fmm_formulation`.
        inverse_z_permeability_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permeability.
        z_permeability_matrix: The Fourier convolution matrix for the z-component
            of the permeability.
        transverse_permeability_matrix: The transverse permittivity matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    if formulation is Formulation.FFT:
        _transverse_permittivity_fn = _transverse_permittivity_fft_anisotropic
        _transverse_permeability_fn = _transverse_permeability_fft_anisotropic
    else:
        vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
        tx, ty = vector_fn(vector_field_source, primitive_lattice_vectors)
        _transverse_permittivity_fn = functools.partial(
            _transverse_permittivity_vector_anisotropic, tx=tx, ty=ty
        )
        _transverse_permeability_fn = functools.partial(
            _transverse_permeability_vector_anisotropic, tx=tx, ty=ty
        )

    (
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
    ) = permittivities
    inverse_z_permittivity_matrix = _transform(1 / permittivity_zz)
    z_permittivity_matrix = _transform(permittivity_zz)
    transverse_permittivity_matrix = _transverse_permittivity_fn(
        permittivity_xx=permittivity_xx,
        permittivity_xy=permittivity_xy,
        permittivity_yx=permittivity_yx,
        permittivity_yy=permittivity_yy,
        expansion=expansion,
    )

    (
        permeability_xx,
        permeability_xy,
        permeability_yx,
        permeability_yy,
        permeability_zz,
    ) = permeabilities
    inverse_z_permeability_matrix = _transform(1 / permeability_zz)
    z_permeability_matrix = _transform(permeability_zz)
    transverse_permeability_matrix = _transverse_permeability_fn(
        permeability_xx=permeability_xx,
        permeability_xy=permeability_xy,
        permeability_yx=permeability_yx,
        permeability_yy=permeability_yy,
        expansion=expansion,
    )
    return (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        inverse_z_permeability_matrix,
        z_permeability_matrix,
        transverse_permeability_matrix,
    )


def _transverse_permittivity_fft_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permittivity matrix for anisotropic media using the fft scheme."""
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permittivity_yy), _transform(-permittivity_yx)],
            [_transform(-permittivity_xy), _transform(permittivity_xx)],
        ]
    )


def _transverse_permeability_fft_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permeability matrix for anisotropic media using the fft scheme."""
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permeability_xx), _transform(permeability_xy)],
            [_transform(permeability_yx), _transform(permeability_yy)],
        ]
    )


def _transverse_permittivity_vector_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permittivity matrix with a vector scheme.

    The transverse permittivity matrix E relates the electric and electric displacement
    fields, such that

        [-Dy, Dx]^T = E [-Ey, Ex]^T

    Args:
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape `(..., nx, ny)`.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The transverse permittivity matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    # Define the real-space and Fourier transformed rotation matrices. The rotation
    # matrix is defined such that T [Et, En]^T = [-Ey, Ex].
    (
        rotation_matrix,
        fourier_rotation_matrix,
        fourier_inverse_rotation_matrix,
    ) = _rotation_matrices(
        t00=ty,
        t01=tx.conj(),
        t10=-tx,
        t11=ty.conj(),
        expansion=expansion,
    )

    # Obtain the permittivity tensor for rotated coordinates, i.e. coordinates where
    # unit vectors are tangent and normal to the vector field defined by `(tx, ty)`.
    permittivity_tensor = jnp.block(
        [
            [
                permittivity_yy[..., jnp.newaxis, jnp.newaxis],
                -permittivity_yx[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                -permittivity_xy[..., jnp.newaxis, jnp.newaxis],
                permittivity_xx[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )
    assert rotation_matrix.shape[-2:] == permittivity_tensor.shape[-2:] == (2, 2)
    rotated_permittivity_tensor = jnp.linalg.solve(
        rotation_matrix, permittivity_tensor @ rotation_matrix
    )

    # The Fourier permittivity matrix is the central matrix in equation 45 of [2012 Liu].
    # The top left block relates the tangential E and D fields, while the bottom right
    # block relates the normal E and D fields. Consequently, these use Laurent's rule and
    # the inverse rule, respectively. Note that the non-diagonal blocks also use Laurent's
    # rule, as these tensor elements may be zero, making the inverse rule problematic.
    fourier_permittivity_matrix = jnp.block(
        [
            [
                _transform(rotated_permittivity_tensor[..., 0, 0]),
                _transform(rotated_permittivity_tensor[..., 0, 1]),
            ],
            [
                _transform(rotated_permittivity_tensor[..., 1, 0]),
                jnp.linalg.inv(_transform(1 / rotated_permittivity_tensor[..., 1, 1])),
            ],
        ]
    )
    return (
        fourier_rotation_matrix
        @ fourier_permittivity_matrix
        @ fourier_inverse_rotation_matrix
    )


def _transverse_permeability_vector_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """Compute the transverse permeability matrix with a vector scheme.

    The transverse permeability matrix M relates the magnetic and magnetic flux
    density fields, such that

        [Bx, Bx]^T = M [Hx, Hy]^T

    Important differences from the `_transverse_permittivity_vector_anisotropic`
    function result from the different definitions of E and M matrices.

    Args:
        permeability_xx: The xx-component of the permeability tensor, with
            shape `(..., nx, ny)`.
        permeability_xy: The xy-component of the permeability tensor.
        permeability_yx: The yx-component of the permeability tensor.
        permeability_yy: The yy-component of the permeability tensor.
        tx: The x-component of the tangent vector field.
        ty: The y-component of the tangent vector field.
        expansion: The field expansion to be used.

    Returns:
        The transverse permeability matrix.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    # Define the real-space and Fourier transformed rotation matrices. The rotation
    # matrix is defined such that T [Ht, Hn]^T = [Hx, Hy], and differs from that
    # used for the transverse permittivity calculation.
    (
        rotation_matrix,
        fourier_rotation_matrix,
        fourier_inverse_rotation_matrix,
    ) = _rotation_matrices(
        t00=tx,
        t01=-ty.conj(),
        t10=ty,
        t11=tx.conj(),
        expansion=expansion,
    )

    # Obtain the permeability tensor for rotated coordinates, i.e. coordinates where
    # unit vectors are tangent and normal to the vector field defined by `(tx, ty)`.
    permeability_tensor = jnp.block(
        [
            [
                permeability_xx[..., jnp.newaxis, jnp.newaxis],
                permeability_xy[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                permeability_yx[..., jnp.newaxis, jnp.newaxis],
                permeability_yy[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )
    assert rotation_matrix.shape[-2:] == permeability_tensor.shape[-2:] == (2, 2)
    rotated_permeability_tensor = jnp.linalg.solve(
        rotation_matrix, permeability_tensor @ rotation_matrix
    )

    # The Fourier permeability matrix is analogous to the central matrix in equation 45
    # of [2012 Liu], in that its top-left block relates the tangential magnetic and
    # magnetic flux density fields, and the bottom-right block relates the normal
    # fields. Consequently, these use Laurent's rule and the inverse rule, respectively.
    # Note that the non-diagonal blocks also use Laurent's rule, as these tensor elements
    # may be zero, making the inverse rule problematic.
    fourier_permeability_matrix = jnp.block(
        [
            [
                _transform(rotated_permeability_tensor[..., 0, 0]),  # tt
                _transform(rotated_permeability_tensor[..., 0, 1]),  # tn
            ],
            [
                _transform(rotated_permeability_tensor[..., 1, 0]),  # nt
                jnp.linalg.inv(
                    _transform(1 / rotated_permeability_tensor[..., 1, 1])
                ),  # nn
            ],
        ]
    )
    return (
        fourier_rotation_matrix
        @ fourier_permeability_matrix
        @ fourier_inverse_rotation_matrix
    )


def _rotation_matrices(
    t00: jnp.ndarray,
    t01: jnp.ndarray,
    t10: jnp.ndarray,
    t11: jnp.ndarray,
    expansion: basis.Expansion,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate real-space and Fourier transformed rotation matrices."""
    rotation_matrix = jnp.block(
        [
            [
                t00[..., jnp.newaxis, jnp.newaxis],
                t01[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                t10[..., jnp.newaxis, jnp.newaxis],
                t11[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )

    # Blockwise Fourier transform of the rotation matrix.
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    fourier_rotation_matrix = jnp.block(
        [
            [_transform(t00), _transform(t01)],
            [_transform(t10), _transform(t11)],
        ]
    )

    # Blockwise Fourier transform of the inverse rotation matrix.
    denom = jnp.abs(t00 * t11 - t10 * t01)
    denom_safe = jnp.where(jnp.isclose(denom, 0), 1.0, denom)
    fourier_inverse_rotation_matrix = jnp.block(
        [
            [_transform(t11 / denom_safe), _transform(-t01 / denom_safe)],
            [_transform(-t10 / denom_safe), _transform(t00 / denom_safe)],
        ]
    )
    return rotation_matrix, fourier_rotation_matrix, fourier_inverse_rotation_matrix


# -----------------------------------------------------------------------------
# Helper functions for validation and matrix assembly.
# -----------------------------------------------------------------------------


def _validate_and_broadcast(
    angular_frequency: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    *permittivities: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """Validates that shapes are compatible and adds required batch dimensions."""
    if not in_plane_wavevector.shape[-1] == 2:
        raise ValueError(
            f"`in_plane_wavevector` must have a final dimension of size 2 but got "
            f"a shape of {in_plane_wavevector.shape}."
        )

    if not all([permittivities[0].shape == p.shape for p in permittivities]):
        raise ValueError("Got permittivities with differing shapes.")

    permittivity = permittivities[0]
    if not utils.batch_compatible_shapes(
        angular_frequency.shape,
        in_plane_wavevector.shape[:-1],
        permittivity.shape[:-2],
    ):
        raise ValueError(
            f"`angular_frequency`, `in_plane_wavevector`, and `permittivity` "
            f"must be batch-compatible, but got shapes of {angular_frequency.shape}, "
            f"{in_plane_wavevector.shape}, and {permittivity.shape}."
        )

    num_batch_dims = max(
        [
            angular_frequency.ndim,
            in_plane_wavevector.ndim - 1,
            permittivity.ndim - 2,
        ]
    )
    angular_frequency = utils.atleast_nd(angular_frequency, n=num_batch_dims)
    in_plane_wavevector = utils.atleast_nd(in_plane_wavevector, n=num_batch_dims + 1)

    permittivities = tuple(
        [utils.atleast_nd(p, n=num_batch_dims + 2) for p in permittivities]
    )
    return (angular_frequency, in_plane_wavevector) + permittivities


def _select_eigenvalues_sign(eigenvalues: jnp.ndarray) -> jnp.ndarray:
    """Selects the sign of eigenvalues to have strictly positive imaginary part.

    Args:
        eigenvalues: The eigenvalues whose sign is to be adjusted.

    Returns:
        The eigenvalues with adjusted sign.
    """
    return jnp.where(jnp.imag(eigenvalues) < 0, -eigenvalues, eigenvalues)


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


jax.tree_util.register_pytree_node(
    Formulation,
    lambda x: ((), x.value),
    lambda value, _: Formulation(value),
)


jax.tree_util.register_pytree_node(
    LayerSolveResult,
    lambda x: (
        (
            x.wavelength,
            x.in_plane_wavevector,
            x.primitive_lattice_vectors,
            x.expansion,
            x.eigenvalues,
            x.eigenvectors,
            x.z_permittivity_matrix,
            x.inverse_z_permittivity_matrix,
            x.z_permeability_matrix,
            x.inverse_z_permeability_matrix,
            x.omega_script_k_matrix,
        ),
        None,
    ),
    lambda _, x: LayerSolveResult(*x),
)
