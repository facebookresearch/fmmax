---
id: utils.box_downsample
---

    
### `utils.box_downsample`
Downsamples `x` to a coarser resolution array using box downsampling.

Box downsampling forms nonoverlapping windows and simply averages the
pixels within each window. For example, downsampling `(0, 1, 2, 3, 4, 5)`
with a factor of `2` yields `(0.5, 2.5, 4.5)`.

#### Args:
- **x** (None): The array to be downsampled.
- **shape** (None): The shape of the output array; each axis dimension must evenly
divide the corresponding axis dimension in `x`.

#### Returns:
- **None**: The output array with shape `shape`.
