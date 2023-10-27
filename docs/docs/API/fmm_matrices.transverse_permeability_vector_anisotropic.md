---
id: fmm_matrices.transverse_permeability_vector_anisotropic
---

    
### `fmm_matrices.transverse_permeability_vector_anisotropic`
Compute the transverse permeability matrix with a vector scheme.

The transverse permeability matrix M relates the magnetic and magnetic flux
density fields, such that

    [Bx, Bx]^T = M [Hx, Hy]^T

Important differences from the `_transverse_permittivity_vector_anisotropic`
function result from the different definitions of E and M matrices.

#### Args:
- **permeability_xx** (None): The xx-component of the permeability tensor, with
shape `(..., nx, ny)`.
- **permeability_xy** (None): The xy-component of the permeability tensor.
- **permeability_yx** (None): The yx-component of the permeability tensor.
- **permeability_yy** (None): The yy-component of the permeability tensor.
- **tx** (None): The x-component of the tangent vector field.
- **ty** (None): The y-component of the tangent vector field.
- **expansion** (None): The field expansion to be used.

#### Returns:
- **None**: The transverse permeability matrix.
