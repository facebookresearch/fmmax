---
id: scattering.stack_s_matrix
---

    
### `scattering.stack_s_matrix`
Computes the s-matrix for a stack of layers.

If only a single layer is provided, the scattering matrix is just the
identity matrix, and start and end layer data is for the same layer.

#### Args:
- **layer_solve_results** (None): The eigensolve results for layers in the stack.
- **layer_thicknesses** (None): The thicknesses for layers in the stack.

#### Returns:
- **None**: The `ScatteringMatrix`.
