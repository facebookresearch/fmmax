---
id: utils.padded_conv
---

    
### `utils.padded_conv`
Convolves `x` with `kernel`, using padding with the specified mode.

Before the convolution, `x` is padded using the specified padding mode.

#### Args:
- **x** (None): The source array.
- **kernel** (None): The rank-2 convolutional kernel.
- **padding_mode** (None): One of the padding modes supported by `jnp.pad`.

#### Returns:
- **None**: The result of the convolution.
