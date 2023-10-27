---
id: vector.change_vector_field_basis
---

    
### `vector.change_vector_field_basis`
Changes the basis for a vector field.

Specifically, given a field with amplitudes `(tu, tv)` of the basis
vectors `(u, v)`, this function computes the amplitudes for the basis
vectors `(x, y)`.

#### Args:
- **tu** (None): The amplitude of the first basis vector in the original basis.
- **tv** (None): The amplitude of the second basis vector in the original basis.
- **u** (None): The first vector of the original basis.
- **v** (None): The second vector of the original basis.
- **x** (None): The first vector of the new basis.
- **y** (None): The second vector of the new basis.

#### Returns:
- **None**: The field `(tx, ty)` in the new basis.
