---
id: fmm.eigensolve_isotropic_media
---

    
### `fmm.eigensolve_isotropic_media`
Performs the eigensolve for a layer with isotropic permittivity.

This function performs either a uniform-layer or patterned-layer eigensolve,
depending on the shape of the trailing dimensions of a given layer permittivity.
When the final two dimensions have shape `(1, 1)`, the layer is treated as
uniform. Otherwise, it is patterned.

#### Args:
- **wavelength** (None): The free space wavelength of the excitation.
- **in_plane_wavevector** (None): `(kx0, ky0)`.
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.
- **permittivity** (None): The permittivity array.
- **expansion** (None): The field expansion to be used.
- **formulation** (None): Specifies the formulation to be used.

#### Returns:
- **None**: The `LayerSolveResult`.
