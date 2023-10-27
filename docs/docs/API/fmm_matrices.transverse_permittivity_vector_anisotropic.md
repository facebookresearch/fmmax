---
id: fmm_matrices.transverse_permittivity_vector_anisotropic
---

    
### `fmm_matrices.transverse_permittivity_vector_anisotropic`
Compute the transverse permittivity matrix with a vector scheme.

The transverse permittivity matrix E relates the electric and electric displacement
fields, such that

    [-Dy, Dx]^T = E [-Ey, Ex]^T

#### Args:
- **permittivity_xx** (None): The xx-component of the permittivity tensor, with
shape `(..., nx, ny)`.
- **permittivity_xy** (None): The xy-component of the permittivity tensor.
- **permittivity_yx** (None): The yx-component of the permittivity tensor.
- **permittivity_yy** (None): The yy-component of the permittivity tensor.
- **tx** (None): The x-component of the tangent vector field.
- **ty** (None): The y-component of the tangent vector field.
- **expansion** (None): The field expansion to be used.

#### Returns:
- **None**: The transverse permittivity matrix.
