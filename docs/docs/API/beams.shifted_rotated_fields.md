---
id: beams.shifted_rotated_fields
---

    
### `beams.shifted_rotated_fields`
Computes the fields on a rotated coordinate system.

Given `fields_fn(xf, yf, zf) -> (exf, eyf, ezf), (hxf, hyf, hzf)` which
returns the fields _in the field coordinate system_, returns the fields
at coordinates `(x, y, z)`, which are rotated from `(xf, yf, zf)`, by
the specified `polar_angle`, `azimuthal_angle`, and `polarization_angle`.

A beam propagating in the `zf` direction, polarized in the `xf` direction
will be propagating in the direction specified by `polar_angle` and
`azimuthal_angle`, with polarization rotated about the propagation
direction by `polarization_angle`.

#### Args:
- **field_fn** (None): Function which returns the fields in the field coordinate
system. The fields should be for a beam propagating in the zf
direction, i.e. in the z-direction of the beam coordinate system.
- **x** (None): x-coordinates of the desired output fields.
- **y** (None): y-coordinates of the desired output fields.
- **z** (None): z-coordinates of the desired output fields.
- **beam_origin_x** (None): The x-origin of the beam coordinate system in the
`(x, y, z)` unit system.
- **beam_origin_y** (None): The y-origin of the beam coordinate system.
- **beam_origin_z** (None): The z-origin of the beam coordinate system.
- **polar_angle** (None): The rotation angle about the y-axis.
- **azimuthal_angle** (None): The rotation angle about the z-axis.
- **polarization_angle** (None): The rotation angle about the propagation axis.

#### Returns:
- **None**: The fields `((ex, ey, ez), (hx, hy, hz))` at the specified coordinates.
