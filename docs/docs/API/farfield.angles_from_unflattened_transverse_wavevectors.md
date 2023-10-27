---
id: farfield.angles_from_unflattened_transverse_wavevectors
---

    
### `farfield.angles_from_unflattened_transverse_wavevectors`
Computes the propagation angles in free space for given wavevectors.

Evanescent modes whose transverse wavevector magnitude exceeds that of
the free space wavevector are given a polar angle of `pi / 2`.

#### Args:
- **transverse_wavevectors** (None): The unflattened transverse wavectors, with
shape `(..., nkx, nky, 2)`.
- **wavelength** (None): The free-space wavelength.

#### Returns:
- **None**: Arrays containing the polar and azimuthal angles.
