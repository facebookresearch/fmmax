---
id: farfield.unflatten_flux
---

    
### `farfield.unflatten_flux`
Unflattens a flux for a given expansion and Brillouin integration scheme.


#### Args:
- **flux** (None): The flux array, with shape `(..., num_bz_kx, num_bz_ky, ...
2 * num_terms, num_sources)`.
- **expansion** (None): The expansion used for the flux.
- **brillouin_grid_axes** (None): The axes associated with the Brillouin zone grid.

#### Returns:
- **None**: The unflattened flux, with shape `(..., num_kx, num_ky, 2, num_sources)`.
