---
id: fields.eigenmode_poynting_flux
---

    
### `fields.eigenmode_poynting_flux`
Returns the total Poynting flux for each eigenmode.

The result is equivalent to summing over the orders of the flux calculated
by `amplitude_poynting_flux`, if the calculation is done for each eigenmode
with a one-hot forward amplitude vector.

#### Args:
- **layer_solve_result** (None): The results of the layer eigensolve.

#### Returns:
- **None**: The per-eigenmode Poynting flux, with the same shape as the eigenvalues.
