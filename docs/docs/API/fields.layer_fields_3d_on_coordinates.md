---
id: fields.layer_fields_3d_on_coordinates
---

    
### `fields.layer_fields_3d_on_coordinates`
Computes the three-dimensional fields in a layer at specified coordinates

This function may be significantly faster than `layer_fields_3d` in cases where
fields in the full simulation domain are not required.

#### Args:
- **forward_amplitude_start** (None): The forward-going wave amplitudes, defined at the
start of the layer.
- **backward_amplitude_end** (None): The backward-going wave amplitudes, defined at the
end of the layer.
- **layer_solve_result** (None): The results of the layer eigensolve.
- **layer_thickness** (None): The layer thickness.
- **layer_znum** (None): The number of gridpoints in the z-direction for the layer.
- **x** (None): The x-coordinates where the fields are sought.
- **y** (None): The y-coordinates where the fields are sought, with shape matching
that of `x`.

#### Returns:
- **None**: The electric and magnetic fields and grid coordinates, `(ef, hf, (x, y, z))`.
