---
id: farfield.unflatten_transverse_wavevectors
---

    
### `farfield.unflatten_transverse_wavevectors`
Unflattens transverse wavevectors for a given expansion and Brillouin integration scheme.


#### Args:
- **transverse_wavevectors** (None): The transverse wavevectors array, with shape
`(..., num_bz_kx, num_bz_ky, ..., num_terms, 2)`.
- **expansion** (None): The expansion used for the flux.
- **brillouin_grid_axes** (None): The axes associated with the Brillouin zone grid.

#### Returns:
- **None**: The unflattened wavevectors, with shape `(..., num_kx, num_ky, 2)`.
