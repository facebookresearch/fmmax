---
id: scattering.append_layer
---

    
### `scattering.append_layer`
Returns new scattering matrix for the stack with an appended layer.


#### Args:
- **s_matrix** (None): The existing scattering matrix.
- **next_layer_solve_result** (None): The eigensolve result for the layer to append.
- **next_layer_thickness** (None): The thickness for the layer to append.

#### Returns:
- **None**: The new `ScatteringMatrix`.
