---
id: fields.fields_from_wave_amplitudes
---

    
### `fields.fields_from_wave_amplitudes`
Computes the electric and magnetic fields inside a layer.

The calculation is for a batch of amplitudes, with the batch axis being the
final axis. There can also be leading batch axes. Accordingly, amplitudes
should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
dimension is preferred because it allows matrix-matrix multiplication instead
of batched matrix-vector multiplication.

#### Args:
- **forward_amplitude** (None): The amplitude of the forward-propagating waves.
- **backward_amplitude** (None): The amplitude of the backward-propagating waves,
at the same location in space as the `forward_amplitude`.
- **layer_solve_result** (None): The results of the layer eigensolve.

#### Returns:
- **None**: The electric and magnetic fields, `((ex, ey, ez), (hx, hy, hz))`.
