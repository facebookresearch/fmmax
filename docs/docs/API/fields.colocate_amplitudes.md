---
id: fields.colocate_amplitudes
---

    
### `fields.colocate_amplitudes`
Compute the forward- and backward-propagating wave amplitudes at `z_offset`.

The calculation is for a batch of amplitudes, with the batch dimension being
final dimension.

#### Args:
- **forward_amplitude_start** (None): The amplitude of the forward eigenmodes at the
start of the layer, with a trailing batch dimension.
- **backward_amplitude_end** (None): The amplitude of the backward eigenmodes at the
end of the layer.
- **z_offset** (None): The location where the colocated amplitudes are sought, as an
offset from the start of the layer.
- **layer_solve_result** (None): The result of the layer eigensolve.
- **layer_thickness** (None): The thickness of the layer.

#### Returns:
- **None**: The forward- and backward-propagating wave amplitudes at `z`.
