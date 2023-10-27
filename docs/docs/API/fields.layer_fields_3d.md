---
id: fields.layer_fields_3d
---

    
### `fields.layer_fields_3d`
Computes the three-dimensional fields in a layer on the real-space grid.


#### Args:
- **forward_amplitude_start** (None): The forward-going wave amplitudes, defined at the
start of the layer.
- **backward_amplitude_end** (None): The backward-going wave amplitudes, defined at the
end of the layer.
- **layer_solve_result** (None): The results of the layer eigensolve.
- **layer_thickness** (None): The layer thickness.
- **layer_znum** (None): The number of gridpoints in the z-direction for the layer.
- **grid_shape** (None): The shape of the xy real-space grid.
- **num_unit_cells** (None): The number of unit cells along each direction.

#### Returns:
- **None**: The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
