---
id: sources.emission_matrix
---

    
### `sources.emission_matrix`
Returns the emission matrix for a source between two layers.


#### Args:
- **s_matrix_before_source** (None): The scattering matrix for the layer substack
before the source, having no overlap with the after-source substack.
Scattering matrix pairs returned by `scattering.stack_s_matrices_interior`
may not be directly used.
- **s_matrix_after_source** (None): The scattering matrix for the layer substack after
the source.

#### Returns:
- **None**: The emission matrix.
