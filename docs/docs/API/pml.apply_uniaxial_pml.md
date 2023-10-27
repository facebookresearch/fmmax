---
id: pml.apply_uniaxial_pml
---

    
### `pml.apply_uniaxial_pml`
Generate the permittivity and permeability tensor elements for uniaxial pml.

The PML assumes that the unit cell has primitive lattice vectors u and v
which are parallel to x and y axes, respectively.

This function is appropriate for isotropic nonmagnetic media, but the
permittivities and permeabilities generated are anisotropic.

#### Args:
- **permittivity** (None): isotropic permittivity
- **pml_params** (None): The parameters defining the perfectly matched layer dimensions
and absorption characteristics.

#### Returns:
- **None**: The permittivity and permeability tensor elements,
`((permittivity_xx, permittivity_xy, permittivity_yx, permittivity_yy, permittivity_zz),
  (permeability_xx, permeability_xy, permeability_yx, permeability_yy, permeability_zz))`.
