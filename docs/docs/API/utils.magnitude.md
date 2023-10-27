---
id: utils.magnitude
---

    
### `utils.magnitude`
Computes elementwise magnitude of the vector field defined by `(tx, ty)`.

This method computes the magnitude with special logic that avoides `nan`
gradients when the magnitude is zero.

#### Args:
- **tx** (None): Array giving the x-component of the vector field.
- **ty** (None): Array giving the y-component of the vector field.

#### Returns:
- **None**: Array giving the vector magnitude.
