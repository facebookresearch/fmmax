---
id: farfield.integrated_flux
---

    
### `farfield.integrated_flux`
Computes the flux within the bounds defined by `angle_bounds_fn`.


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
- **angle_bounds_fn** (None): A function with signature `fn(polar_angle, azimuthal_angle)`
returning a mask that is `True` for angles that should be included in
the integral.
- **upsample_factor** (None): Integer factor specifying upsampling performed in the
integral, which is used to approximate trapezoidal rule integration.

#### Returns:
- **None**: The integrated flux, with shape equal to the batch dimensions of flux,
excluding those for the brillouin zone grid.
