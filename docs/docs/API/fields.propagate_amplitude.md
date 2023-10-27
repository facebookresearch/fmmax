---
id: fields.propagate_amplitude
---

    
### `fields.propagate_amplitude`
Propagates waves with the given `amplitude` by `distance`.

The propagation is along the wave direction, i.e. when `distance` is positive,
amplitudes for forward-propagating waves are those associated with a positive
shift along the z-axis, while the reverse is true for backward-propagating wave
amplitudes.

#### Args:
- **amplitude** (None): The amplitudes to be propagated, with a trailing batch dimension.
- **distance** (None): The distance to be propagated.
- **layer_solve_result** (None): The result of the layer eigensolve.

#### Returns:
- **None**: The propagated amplitudes.
