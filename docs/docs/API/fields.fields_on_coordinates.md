---
id: fields.fields_on_coordinates
---

    
### `fields.fields_on_coordinates`
Computes the fields at specified coordinates.

The calculation is for a batch of fields, with the batch axis being the
final axis. There can also be leading batch axes. Accordingly, fields
should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
dimension is preferred because it allows matrix-matrix multiplication instead
of batched matrix-vector multiplication.

#### Args:
- **electric_field** (None): `(ex, ey, ez)` electric field Fourier amplitudes.
- **magnetic_field** (None): `(hx, hy, hz)` magnetic field Fourier amplitudes.
- **layer_solve_result** (None): The results of the layer eigensolve.
- **x** (None): The x-coordinates where the fields are sought.
- **y** (None): The y-coordinates where the fields are sought, with shape matching
that of `x`.

#### Returns:
- **None**: The electric field `(ex, ey, ez)`, magnetic field `(hx, hy, hz)`,
and the grid coordinates `(x, y)`. The field arrays each have shape
`batch_shape + coordinates_shape + (num_amplitudes)`.
