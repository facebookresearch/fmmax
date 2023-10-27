---
id: fields.directional_poynting_flux
---

    
### `fields.directional_poynting_flux`
Returns total forward and backward Poynting flux.

This function decomposes the total field into components resulting from the
the eigenmodes with positive and negative Poynting flux, and returns the
time-average flux in each order for these two components. The calculation
follows section 5.1 of [2012 Liu].

In the general case, a forward eigenmode may actually have negative
Poynting flux, and so e.g. it may occur that a one-hot forward amplitude
vector yields zero forward flux and nonzero backward flux.

If the flux associated with the forward and backward eigenmodes is desired,
`amplitude_poynting_flux` should be used instead. This function serves the
more typical case where the total forward flux and total backward flux is
desired.

#### Args:
- **forward_amplitude** (None): The amplitude of the forward eigenmodes, with a
trailing batch dimension.
- **backward_amplitude** (None): The amplitude of the backward eigenmodes, at the
same location in space as the `forward_amplitude`.
- **layer_solve_result** (None): The results of the layer eigensolve.

#### Returns:
- **None**: The Poynting flux associated with the forward and backward eigenmodes.
