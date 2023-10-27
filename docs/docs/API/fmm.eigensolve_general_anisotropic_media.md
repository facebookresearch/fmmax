---
id: fmm.eigensolve_general_anisotropic_media
---

    
### `fmm.eigensolve_general_anisotropic_media`
Performs the eigensolve for a general anistropic layer.

Here, "general" refers to the fact that the layer material can be magnetic, i.e.
the permeability and permittivity can be specified.

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
- **permeability_xx** (None): The xx-component of the permeability tensor.
- **permeability_xy** (None): The xy-component of the permeability tensor.
- **permeability_yx** (None): The yx-component of the permeability tensor.
- **permeability_yy** (None): The yy-component of the permeability tensor.
- **permeability_zz** (None): The zz-component of the permeability tensor.
- **expansion** (None): The field expansion to be used.
- **formulation** (None): Specifies the formulation to be used.
- **vector_field_source** (None): Optional array used to calculate the vector field for
vector formulations of the FMM. If not specified, `(permittivity_xx +
permittivity_yy) / 2` is used. Ignored for the `FFT` formulation. Should
have shape matching the permittivities and permeabilities.

#### Returns:
- **None**: The `LayerSolveResult`.
