---
id: fmm.fourier_matrices_patterned_isotropic_media
---

    
### `fmm.fourier_matrices_patterned_isotropic_media`
Return Fourier convolution matrices for patterned nonmagnetic isotropic media.

All matrices are forms of the Fourier convolution matrices defined in equation
8 of [2012 Liu]. For vector formulations, the transverse permittivity matrix is
of the form E2 given in equation 51 of [2012 Liu].

#### Args:
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.
- **permittivity** (None): The permittivity array, with shape `(..., nx, ny)`.
- **expansion** (None): The field expansion to be used.
- **formulation** (None): Specifies the formulation to be used, e.g. a vector formulation
or the non-vector `FFT` formulation.

#### Returns:
- **inverse_z_permittivity_matrix**: The Fourier convolution matrix for the inverse
of the z-component of the permittivity.
z_permittivity_matrix: The Fourier convolution matrix for the z-component
    of the permittivity.
transverse_permittivity_matrix: The transverse permittivity matrix.
