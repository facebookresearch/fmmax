---
id: utils.eig
---

    
### `utils.eig`
Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

The custom vjp allows gradients with resepct to the eigenvectors, unlike the
standard jax implementation of `eig`. We use an expression for the gradient
given in [2019 Boeddeker] along with a regularization scheme used in [2021
Colburn]. The method effectively applies a Lorentzian broadening to a term
containing the inverse difference of eigenvalues.

[2019 Boeddeker] https://arxiv.org/abs/1701.00392
[2021 Coluburn] https://www.nature.com/articles/s42005-021-00568-6

#### Args:
- **matrix** (None): The matrix for which eigenvalues and eigenvectors are sought.
- **eps** (None): Parameter which determines the degree of broadening.

#### Returns:
- **None**: The eigenvalues and eigenvectors.
