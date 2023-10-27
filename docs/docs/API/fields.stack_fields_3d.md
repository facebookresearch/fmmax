---
id: fields.stack_fields_3d
---

    
### `fields.stack_fields_3d`
Computes the three-dimensional fields in a stack on the real-space grid.


#### Args:
- **amplitudes_interior** (None): The forward- and backward-propagating wave amplitude
for each layer, defined at the start and end of each layer, respectively.
- **layer_solve_results** (None): The results of the layer eigensolve for each layer.
- **layer_thicknesses** (None): The thickness of each layer.
- **layer_znum** (None): The number of gridpoints in the z-direction for each layer.
- **grid_shape** (None): The shape of the xy real-space grid.
- **num_unit_cells** (None): The number of unit cells along each direction.

#### Returns:
- **None**: The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
