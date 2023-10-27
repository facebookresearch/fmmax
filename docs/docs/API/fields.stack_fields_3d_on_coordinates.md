---
id: fields.stack_fields_3d_on_coordinates
---

    
### `fields.stack_fields_3d_on_coordinates`
Computes the three-dimensional fields in a stack at specified coordinates.

This function may be significantly faster than `stack_fields_3d` in cases where
fields in the full simulation domain are not required.

#### Args:
- **amplitudes_interior** (None): The forward- and backward-propagating wave amplitude
for each layer, defined at the start and end of each layer, respectively.
- **layer_solve_results** (None): The results of the layer eigensolve for each layer.
- **layer_thicknesses** (None): The thickness of each layer.
- **layer_znum** (None): The number of gridpoints in the z-direction for each layer.
- **x** (None): The x-coordinates where the fields are sought.
- **y** (None): The y-coordinates where the fields are sought, with shape matching
that of `x`.

#### Returns:
- **None**: The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
