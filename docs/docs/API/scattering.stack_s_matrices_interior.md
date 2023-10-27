---
id: scattering.stack_s_matrices_interior
---

    
### `scattering.stack_s_matrices_interior`
Computes scattering matrices before and after each layer in the stack.

Specifically, for each layer `i` two `ScatteringMatrix` are returned. The
first relates fields in the substack `0, ..., i`, while the second relates
the fields in the substack `i, ..., N`, where `N` is the maximum layer
index. These two scattering matrices are denoted as the corresponding
to the "before" substack and the "after" substack.

#### Args:
- **layer_solve_results** (None): The eigensolve results for layers in the stack.
- **layer_thicknesses** (None): The thicknesses for layers in the stack.

#### Returns:
- **None**: The tuple of `(scattering_matrix_before, scattering_matrix_after)`.
