---
id: vector.compute_gradient
---

    
### `vector.compute_gradient`
Computes the gradient of `arr` with respect to `x` and `y`.

This function uses periodic boundary conditions. The `x` and `y`
dimensions correspond to the trailing axes of `arr`.

#### Args:
- **arr** (None): The array whose gradient is sought.

#### Returns:
- **None**: The `(gx, gy)` gradients along the x- and y- directions.
