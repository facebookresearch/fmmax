---
id: fmm_matrices.transverse_permittivity_vector
---

    
### `fmm_matrices.transverse_permittivity_vector`
Computes transverse permittivity matrix using a vector field methods.

The transverse permittivity matrix is E2 given in equation 51 of [2012 Liu].

#### Args:
- **permittivity** (None): The permittivity array, with shape `(..., nx, ny)`.
- **tx** (None): The x-component of the tangent vector field.
- **ty** (None): The y-component of the tangent vector field.
- **expansion** (None): The field expansion to be used.

#### Returns:
- **None**: The `eps` matrix.
