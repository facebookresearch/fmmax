---
id: vector.tangent_field
---

    
### `vector.tangent_field`
Computes a real or complex tangent vector field.

The field is tangent to the interfaces of features in `arr`, and varies smoothly
between interfaces. It is obtained by optimization which minimizes a functional
of the field which favors alignment with interfaces of features in `arr` as well
as smoothness of the field. The maximum magnitude of the computed field is `1`.

The tangent field is complex when `use_jones` is `True`, and real otherwise. Real
fields are suitable for normalization using methods in this module. The complex
field obtained when `use_jones` is `True` requires no normalization.

#### Args:
- **arr** (None): The array for which the tangent field is sought.
- **use_jones** (None): Specifies whether a complex Jones field or a real tangent vector
field is sought.
- **optimizer** (None): The optimizer used to minimize the functional.
- **alignment_weight** (None): The weight of an alignment term in the functional. Larger
values will reward alignment with interfaces of features in `arr`.
- **smoothness_weight** (None): The weight of a smoothness term in the functional. Larger
values will reward a smoother tangent field.
- **steps_dim_multiple** (None): Controls the number of steps in the optimization. The
number of steps is the product of `steps_dim_multiple` and the dimension
of the largest of the two trailing axes of `arr`.
- **smoothing_kernel** (None): Kernel used to smooth `arr` prior to the computation.

#### Returns:
- **None**: The tangent vector fields, `(tx, ty)`.
