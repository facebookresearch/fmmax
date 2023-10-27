---
id: utils.interpolate_permittivity
---

    
### `utils.interpolate_permittivity`
Interpolates the permittivity with a scheme that avoids zero crossings.

The interpolation uses the scheme introduced in [2019 Christiansen], which avoids
zero crossings that can occur with metals or lossy materials having a negative
real component of the permittivity. https://doi.org/10.1016/j.cma.2018.08.034

#### Args:
- **permittivity_solid** (None): The permittivity of solid regions.
- **permittivity_void** (None): The permittivity of void regions.
- **density** (None): The density, specifying which locations are solid and which are void.

#### Returns:
- **None**: The interpolated permittivity.
