---
id: vector.normalized_vector_field
---

    
### `vector.normalized_vector_field`
Generates a normalized tangent vector field according to the specified method.

Some `vector_fn` can be computationally expensive, and so this function
resizes the input `arr` so that the maximum size of the two trailing
dimensions is `resize_max_dim`. When `arr` is smaller than the maximum
size, no resampling is performed.

The tangent fields are then computed for this resized array, and then
resized again to obtain fields at the original resolution.

#### Args:
- **arr** (None): The array for which the tangent vector field is sought.
- **primitive_lattice_vectors** (None): Define the unit cell coordinates.
- **vector_fn** (None): Function used to generate the vector field.
- **normalize_fn** (None): Function used to normalize the vector field.
- **resize_max_dim** (None): Determines the size of the array for which the tangent
vector field is computed; `arr` is resized so that it has a maximum
size of `vector_arr_size` along any dimension.
- **resize_method** (None): Method used in scaling `arr` prior to calculating the
tangent vector field.

#### Returns:
- **None**: The normalized vector field.
