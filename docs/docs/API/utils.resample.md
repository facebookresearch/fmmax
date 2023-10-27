---
id: utils.resample
---

    
### `utils.resample`
Resamples `x` to have the specified `shape`.

The algorithm first upsamples `x` so that the pixels in the output image are
comprised of an integer number of pixels in the upsampled `x`, and then
performs box downsampling.

#### Args:
- **x** (None): The array to be resampled.
- **shape** (None): The shape of the output array.
- **method** (None): The method used to resize `x` prior to box downsampling.

#### Returns:
- **None**: The resampled array.
