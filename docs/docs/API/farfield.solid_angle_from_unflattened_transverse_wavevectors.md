---
id: farfield.solid_angle_from_unflattened_transverse_wavevectors
---

    
### `farfield.solid_angle_from_unflattened_transverse_wavevectors`
Computes the solid angle associated with each transverse wavevector.

The transverse wavevectors should be unflattened, i.e. the `-3` and `-2`
axes should correspond to different points in k-space.

#### Args:
- **transverse_wavevectors** (None): The unflattened transverse wavevectors
- **wavelength** (None): The free-space wavelength.

#### Returns:
- **None**: The solid angle, with the shape matching the leading dimensions of
`transverse_wavevectors`.
