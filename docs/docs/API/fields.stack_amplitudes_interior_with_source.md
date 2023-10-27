---
id: fields.stack_amplitudes_interior_with_source
---

    
### `fields.stack_amplitudes_interior_with_source`
Computes the wave amplitudes in the case of an internal source.


#### Args:
- **s_matrices_interior_before_source** (None): The interior scattering matrices for
the layer substrack before the source, as computed by
`stack_s_matrices_interior`.
- **s_matrices_interior_after_source** (None): The interior scattering matrices for
the layer substack after the source.
- **backward_amplitude_before_end** (None): The backward-going wave amplitude at the
end of the layer before the source.
- **forward_amplitude_after_start** (None): The forward-going wave amplitude at the
start of the layer after the source.

#### Returns:
- **None**: The forward- and backward-propagating wave amplitude for each layer,
defined at the start and end of each layer, respectively.
