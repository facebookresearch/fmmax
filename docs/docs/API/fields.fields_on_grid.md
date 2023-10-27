---
id: fields.fields_on_grid
---

    
### `fields.fields_on_grid`
Transforms the fields from fourier representation to the grid.

The fields within an array of unit cells is returned, with the number of
cells in each direction given by `num_unit_cells`.

The calculation is for a batch of fields, with the batch axis being the
final axis. There can also be leading batch axes. Accordingly, fields
should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
dimension is preferred because it allows matrix-matrix multiplication instead
of batched matrix-vector multiplication.

#### Args:
- **electric_field** (None): `(ex, ey, ez)` electric field Fourier amplitudes.
- **magnetic_field** (None): `(hx, hy, hz)` magnetic field Fourier amplitudes.
- **layer_solve_result** (None): The results of the layer eigensolve.
- **shape** (None): The shape of the grid.
- **num_unit_cells** (None): The number of unit cells along each direction.

#### Returns:
- **None**: The electric field `(ex, ey, ez)`, magnetic field `(hx, hy, hz)`,
and the grid coordinates `(x, y)`.
