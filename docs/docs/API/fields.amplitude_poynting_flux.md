---
id: fields.amplitude_poynting_flux
---

    
### `fields.amplitude_poynting_flux`
Returns total Poynting flux for forward and backward eigenmodes.

This function decomposes the total field into components associated with
the forward and backward amplitudes, and returns the time-average flux in
each order for these two components. The calculation follows section 5.1
of [2012 Liu].

In the general case, a forward eigenmode may actually have negative
Poynting flux, and therefore the quantities computed by this function
should not be interpreted as the total forward and backward flux, but only
the total flux associated with the forward and backward eigenmodes.

If the total forward and backward flux is desired, `directional_poynting_flux`
should be used instead. This function should only be used in the specific
case where the flux associated with the forward and backward eigenmodes is
needed.

#### Args:
- **forward_amplitude** (None): The amplitude of the forward eigenmodes, with a
trailing batch dimension.
- **backward_amplitude** (None): The amplitude of the backward eigenmodes, at the
same location in space as the `forward_amplitude`.
- **layer_solve_result** (None): The results of the layer eigensolve.

#### Returns:
- **None**: The Poynting flux associated with the forward and backward eigenmodes.
