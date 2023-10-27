---
id: fields.stack_amplitudes_interior
---

    
### `fields.stack_amplitudes_interior`
Computes the wave amplitudes at interior layers within a stack.

The calculation is for a batch of amplitudes, with the batch axis being the
final axis. There can also be leading batch axes. Accordingly, amplitudes
should have shape `(..., 2 * num_terms, num_amplitudes)`. The trailing batch
dimension is preferred because it allows matrix-matrix multiplication instead
of batched matrix-vector multiplication.

#### Args:
- **s_matrices_interior** (None): The scattering matrices for the substacks before
and after each layer, as computed by `stack_s_matrices_interior`.
- **forward_amplitude_0_start** (None): The forward-propagating wave amplitude at the
start of the first layer of the stack.
- **backward_amplitude_N_end** (None): The backward-propagating wave amplitude at the
end of the last layer of the stack.

#### Returns:
- **None**: The forward- and backward-propagating wave amplitude for each layer,
defined at the start and end of each layer, respectively.
