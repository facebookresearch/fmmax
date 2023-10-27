---
id: sources.polarization_terms
---

    
### `sources.polarization_terms`
Computes the polarization terms for currents on the real-space grid.

The polarization terms are discussed in section 7 of [1999 Whittaker].

#### Args:
- **jx** (None): The Fourier-transformed x-oriented dipole amplitude, with a
trailing batch dimension.
- **jy** (None): The Fourier-transformed y-oriented dipole amplitude.
- **jz** (None): The Fourier-transformed z-oriented dipole amplitude.
- **layer_solve_result** (None): The results of the layer eigensolve.

#### Returns:
- **None**: The polarization vector containing parallel and z-oriented terms.
