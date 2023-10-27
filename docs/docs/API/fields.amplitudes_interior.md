---
id: fields.amplitudes_interior
---

    
### `fields.amplitudes_interior`
Computes the wave amplitudes at an interior layer within a stack.

The calculation is for a batch of amplitudes, with the batch axis being the
final axis. There can also be leading batch axes. Accordingly, amplitudes
should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
dimension is preferred because it allows matrix-matrix multiplication instead
of batched matrix-vector multiplication.

#### Args:
- **s_matrix_before** (None): The scattering matrix for the substack before the layer.
- **s_matrix_after** (None): The scattering matrix for the substack after the layer.
- **forward_amplitude_0_start** (None): The forward-propagating wave amplitude at the
start of the first layer of the stack.
- **backward_amplitude_N_end** (None): The backward-propagating wave amplitude at the
end of the last layer of the stack.

#### Returns:
- **None**: The forward- and backward-propagating wave amplitude in the layer, defined
at layer start and end, respectively.
