---
id: fmm.fourier_matrices_patterned_anisotropic_media
---

    
### `fmm.fourier_matrices_patterned_anisotropic_media`
Return Fourier convolution matrices for patterned anisotropic media.

The transverse permittivity matrix E is defined as,

    [-Dy, Dx]^T = E [-Ey, Ex]^T

while the transverse permeability matrix M is defined as,

    [Bx, By]^T = M [Hx, Hy]^T

The Fourier factorization is done as for E1 given in equation 47 of [2012 Liu].

#### Args:
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.
- **permittivities** (None): The elements of the permittivity tensor: `(eps_xx, eps_xy,
eps_yx, eps_yy, eps_zz)`, each having shape `(..., nx, ny)`.
- **permeabilities** (None): The elements of the permeability tensor: `(mu_xx, mu_xy,
mu_yx, mu_yy, mu_zz)`, each having shape `(..., nx, ny)`.
- **expansion** (None): The field expansion to be used.
- **formulation** (None): Specifies the formulation to be used.
- **vector_field_source** (None): Array used to calculate the vector field, with shape
matching the permittivities and permeabilities.

#### Returns:
- **inverse_z_permittivity_matrix**: The Fourier convolution matrix for the inverse
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
