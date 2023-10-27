---
id: basis.generate_expansion
---

    
### `basis.generate_expansion`
Generates the expansion for the specified real-space basis.


#### Args:
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.
- **approximate_num_terms** (None): The approximate number of terms in the expansion. To
maintain a symmetric expansion, the total number of terms may differ from
this value.
- **truncation** (None): The truncation to be used for the expansion.

#### Returns:
- **None**: The `Expansion`. The basis coefficients of the expansion are sorted so that
the zeroth-order term is first.
