---
id: fmm_matrices.transverse_permittivity_fft
---

    
### `fmm_matrices.transverse_permittivity_fft`
Computes the `eps` matrix from [2012 Liu] equation 15 using `fft` scheme.


#### Args:
- **permittivity** (None): The permittivity array, with shape `(..., nx, ny)`.
- **expansion** (None): The field expansion to be used.

#### Returns:
- **None**: The transverse permittivity matrix.
