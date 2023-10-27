---
id: fft.fourier_convolution_matrix
---

    
### `fft.fourier_convolution_matrix`
Computes the Fourier convolution matrix for `x` and `basis_coefficients`.

The Fourier convolution matrix at location `(i, j)` gives the Fourier
coefficient associated with the lattice vector obtained by subtracting the
`j`th reciprocal lattice vector from the `i`th reciprocal lattice basis.
See equation 8 from [2012 Liu].

#### Args:
- **x** (None): The array for which the Fourier coefficients are sought.
- **expansion** (None): The field expansion to be used.

#### Returns:
- **None**: The coefficients, with shape `(num_vectors, num_vectors)`.
