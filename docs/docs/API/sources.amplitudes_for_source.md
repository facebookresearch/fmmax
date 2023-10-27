---
id: sources.amplitudes_for_source
---

    
### `sources.amplitudes_for_source`
Computes wave amplitudes resulting from an internal source.

The configuration of the calculation is depicted below. A source is located at
the interface of two layers. The layer stacks before and after the source are
named as such. The function computes the amplitudes of forward-going and
backward-going waves at various locations within the stack, also depicted below.

                            source _____
                                        |
         __before_______________________V___after_____________________

        |             |     |           |           |     |           |
        |   layer 0   | ... |  layer i  |   layer   | ... |  layer N  |
        | start layer |     |           |   i + 1   |     | end layer |
        |             |     |           |           |     |           |

                             -> a_i      -> a_i+1          -> a_N
                b_0 <-            b_i <-    b_i+1 <-

#### Args:
- **jx** (None): The x-oriented dipole amplitude; must be at least rank-3 with a
trailing batch dimension.
- **jy** (None): The y-oriented dipole amplitude, with shape matching `jx`.
- **jz** (None): The z-oriented dipole amplitude, with shape matching `jx`.
- **s_matrix_before_source** (None): The scattering matrix for the layer substack
before the source, having no overlap with the after-source substack.
Scattering matrix pairs returned by `scattering.stack_s_matrices_interior`
may not be directly used.
- **s_matrix_after_source** (None): The scattering matrix for the layer substack after
the source.

#### Returns:
- **None**: The wave amplitudes:
backward_amplitude_0_end: The backward-going wave amplitude at the end
    of the first layer.
forward_amplitude_before_start: The forward-going wave amplitude at the
    start of the layer preceding the source.
backward_amplitude_before_end: The backward-going wave amplitude at the
    end of the layer preceding the source, i.e. just before the source.
forward_amplitude_after_start: The forward-going wave amplitude at the
    start of the layer following the source, i.e. just after the source.
backward_amplitude_after_end: The backward-going wave amplitude at the
    end of the layer following the source.
forward_amplitude_N_start: The forward-going wave amplitude at the start
    of the final layer.
