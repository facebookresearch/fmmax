---
id: basis.brillouin_zone_in_plane_wavevector
---

    
### `basis.brillouin_zone_in_plane_wavevector`
Computes in-plane wavevectors on a regular grid in the first Brillouin zone.

The wavevectors are found by dividing the Brillouin zone into a grid with the
specified shape; the wavevectors are at the centers of the grid voxels.

#### Args:
- **brillouin_grid_shape** (None): The shape of the wavevector grid.
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.

#### Returns:
- **None**: The in-plane wavevectors, with shape `brillouin_grid_shape + (2,)`.
