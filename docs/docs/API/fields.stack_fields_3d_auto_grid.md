---
id: fields.stack_fields_3d_auto_grid
---

    
### `fields.stack_fields_3d_auto_grid`
Computes the three-dimensional fields in a stack on the real-space grid.

The grid is automatically determined from the layer dimensions and the resolution.

#### Args:
- **amplitudes_interior** (None): The forward- and backward-propagating wave amplitude
for each layer, defined at the start and end of each layer, respectively.
- **layer_solve_results** (None): The results of the layer eigensolve for each layer.
- **layer_thicknesses** (None): The thickness of each layer.
- **grid_spacing** (None): The approximate spacing of gridpoints on which the field is
computed. The actual grid spacing is modified to align with the layer
and unit cell boundaries.
- **num_unit_cells** (None): The number of unit cells along each direction.

#### Returns:
- **None**: The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
