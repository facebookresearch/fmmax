---
id: sources.gaussian_source
---

    
### `sources.gaussian_source`
Returns the coefficients for a Gaussian source at the specified location.

This function is appropriate for creating sources to be used with
`amplitudes_for_source`.

#### Args:
- **fwhm** (None): The full-width at half-maximum for the Gaussian source.
- **location** (None): The location of the source, with shape `(num_sources, 2)` and
the trailing axis giving the x and y location. By convention, the
center of the unit cell is at `(0, 0)`.
- **in_plane_wavevector** (None): The in-plane wavevevector for the calculation, which
gives the offset of the plane wave decomposition. Has shape `(..., 2)`
with possible batch dimensions.
- **primitive_lattice_vectors** (None): The primitive lattice vectors of the unit cell.
- **expansion** (None): The Fourier expansion used for the calculation.

#### Returns:
- **None**: The coefficients, with the shape `(..., expansion.num_terms, num_sources)`.
