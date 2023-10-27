---
id: farfield.unflatten
---

    
### `farfield.unflatten`
Unflattens an array for a given expansion and Brillouin integration scheme.

The returned array combines the values associated with all terms in the
Fourier expansion at all points in the Brillouin zone grid in a single
array with trailing axes havving shape `(num_kx, num_ky)`. Elements in the
output which have no corresponding elements in `flat` are given a value
of `nan`.

The flat array should have shape `(..., num_bz_kx, num_bz_ky, num_terms)`,
where `num_terms` is the number of terms in the Fourier expansion, and the
`-3` and `-2` axes are for the Brillouin zone grid, as used e.g. with
Brillouin zone integration to model localized sources.

This function assumes that the Brillouin zone is sampled on a regular grid,
as produced by `basis.brillouin_zone_in_plane_wavevector`.

#### Args:
- **flat** (None): The flat array, with shape  `(..., num_bz_kx, num_bz_ky, num_terms)`.
- **expansion** (None): The expansion used for the array.

#### Returns:
- **None**: The unflattened array, with shape `(batch_shape, num_kx, num_ky)`.
