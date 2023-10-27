---
id: beams.rotation_matrix
---

    
### `beams.rotation_matrix`
Computes a rotation matrix that arbitrarily reorients a field.

The rotation operations consist of,
  - rotation about the y-axis by `polar_angle`
  - rotation about the z-axis by `azimuthal_angle`
  - rotation about propagation axis by `polarization_angle`, where
    the propagation axis is found by applying the first two rotations
    about the y- and z-axis.

#### Args:
- **polar_angle** (None): The rotation angle about the y-axis.
- **azimuthal_angle** (None): The rotation angle about the z-axis.
- **polarization_angle** (None): The rotation angle about the propagation axis.

#### Returns:
- **None**: The rotation matrix.
