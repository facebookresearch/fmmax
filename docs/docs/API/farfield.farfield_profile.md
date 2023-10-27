---
id: farfield.farfield_profile
---

    
### `farfield.farfield_profile`
Computes a farfield profile.

This function effectively "unstacks" the values for each Fourier order and
for each point in the Brillouin zone sampling scheme.

#### Args:
- **flux** (None): The flux array, with shape `(..., num_bz_kx, num_bz_ky, ...
2 * num_terms, num_sources)`.
- **wavelength** (None): The wavelength, batch-compatible with `flux`.
- **in_plane_wavevector** (None): The in-plane wavevector for the zeroth Fourier
order, batch-compatible with `flux`.
- **primitive_lattice_vectors** (None): The primitive lattice vectors of the unit cell.
- **expansion** (None): The expansion used for the fields.
- **brillouin_grid_axes** (None): Specifies the two axes of `flux` corresponding to
the Brillouin zone grid.

#### Returns:
- **None**: The polar and azimuthal angles, solid angle associated with each value,
and the farfield power.
