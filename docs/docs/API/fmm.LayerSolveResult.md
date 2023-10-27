---
id: fmm.LayerSolveResult
---

    ### `Class fmm.LayerSolveResult():`
Stores the result of a layer eigensolve.

This eigenvalue problem is specified in equation 28 of [2012 Liu].

#### Args:
- **wavelength** (None): The wavelength for the solve.
- **in_plane_wavevector** (None): The in-plane wavevector for the solve.
- **primitive_lattice_vectors** (None): The primitive vectors for the real-space lattice.
- **expansion** (None): The expansion used for the eigensolve.
- **eigenvalues** (None): The layer eigenvalues.
- **eigenvectors** (None): The layer eigenvectors.
- **z_permittivity_matrix** (None): The fourier-transformed zz-component of permittivity.
- **inverse_z_permittivity_matrix** (None): The fourier-transformed inverse of zz-component
of permittivity.
- **z_permeability_matrix** (None): The fourier-transformed zz-component of permeability.
- **inverse_z_permeability_matrix** (None): The fourier-transformed inverse of zz-component
of permeability.
- **omega_script_k_matrix** (None): The omega-script-k matrix from equation 26 of
[2012 Liu], which is needed to generate the layer scattering matrix.

