---
id: fmm.eigensolve_anisotropic_media
---

    
### `fmm.eigensolve_anisotropic_media`
Performs the eigensolve for a layer with anisotropic permittivity.

This function performs either a uniform-layer or patterned-layer eigensolve,
depending on the shape of the trailing dimensions of a given layer permittivity.
When the final two dimensions have shape `(1, 1)`, the layer is treated as
uniform. Otherwise, it is patterned.

#### Args:
- **wavelength** (None): The free space wavelength of the excitation.
- **in_plane_wavevector** (None): `(kx0, ky0)`.
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.
- **permittivity_xx** (None): The xx-component of the permittivity tensor, with
shape `(..., nx, ny)`.
- **permittivity_xy** (None): The xy-component of the permittivity tensor.
- **permittivity_yx** (None): The yx-component of the permittivity tensor.
- **permittivity_yy** (None): The yy-component of the permittivity tensor.
- **permittivity_zz** (None): The zz-component of the permittivity tensor.
- **expansion** (None): The field expansion to be used.
- **formulation** (None): Specifies the formulation to be used.

#### Returns:
- **None**: The `LayerSolveResult`.
