---
id: scattering.prepend_layer
---

    
### `scattering.prepend_layer`
Returns new scattering matrix for the stack with a prepended layer.


#### Args:
- **s_matrix** (None): The existing scattering matrix.
- **next_layer_solve_result** (None): The eigensolve result for the layer to append.
- **next_layer_thickness** (None): The thickness for the layer to append.

#### Returns:
- **None**: The new `ScatteringMatrix`.
