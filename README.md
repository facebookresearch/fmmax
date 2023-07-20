# FMMAX: Fourier Modal Method with Jax

FMMAX is a an implementation of the Fourier modal method (FMM) in [JAX](https://github.com/google/jax). 

The FMM -- also known as rigorous coupled wave analysis (RCWA) -- is a semianalytical method that solves Maxwell's equations in periodic stratified media, where in-plane directions are treated with a truncated Fourier basis and the normal direction is handled by a scattering matrix approach. This allows certain classes of structures to be modeled with relatively low computational cost.

FMMAX includes vector formulations of the FMM, which dramatically improve convergence for structures having large permittivity contrast. It also allows for Brillouin zone integration to model isolated sources in periodic structures. Finally, the use of JAX enables GPU acceleration and automatic differentiation of FMM simulations.

The main implementation -- including the _pol_, _normal_, and _jones_ vector FMM formulations -- follows the derivations in [2012 Liu], which is also the basis of the popular [S4](https://web.stanford.edu/group/fan/S4/) code. FMMAX also includes a new _jones direct_ formulation which improves convergence in some cases. Scattering matrix calculations follows the method of [1999 Whittaker]. Another helpful reference is the [grcwa](https://github.com/weiliangjinca/grcwa) code associated with \[2020 Jin], which is an autograd port of S4 and is used as a reference in some tests.

## Conventions
- The speed of light, vacuum permittivity, and vacuum permeability are all 1.
- Fields evolve in time as $\exp(-i \omega t)$.
- If $\mathbf{u}$ and $\mathbf{v}$ are the primitive lattice vectors, the unit cell is defined by the parallelogram with vertices at $\mathbf{0}$, $\mathbf{u}$, $\mathbf{u} + \mathbf{v}$, and $\mathbf{v}$.
- For quantities defined on a grid (such as the permittivity distribution of a patterned layer) the value at grid index (0, 0) corresponds to the value at physical location $\mathbf{0}$.
- Following the convention of [1999 Whittaker], the scattering matrix block $\mathbf{S}_{11}$ relates incident and transmitted forward-going fields, and other blocks have corresponding definitions. This differs from the convention e.g. in photonic integrated circuits.

## Batching
Batched calculations are supported, and should be used where possible to avoid looping. The batch axes are the leading axes, except for the excitation, where a trailing batch axis is assumed. This allows e.g. computing the transmission through a structure for multiple polarizations via a matrix-matrix operation (`transmitted_amplitudes = S11 @ incident_amplitudes`), rather than a batched matrix-vector operation.

## References
- [2012 Liu] V. Liu and S. Fan, [S4: A free electromagnetic solver for layered structures structures](https://www.sciencedirect.com/science/article/pii/S0010465512001658), _Comput. Phys. Commun._ **183**, 2233-2244 (2012).

- [1999 Whittaker] D. M. Whittaker and I. S. Culshaw, [Scattering-matrix treatment of patterned multilayer photonic structures](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.60.2610), _Phys. Rev. B_ **60**, 2610 (1999).

- [2020 Jin] W. Jin, W. Li, M. Orenstein, and S. Fan [Inverse design of lightweight broadband reflector for relativistic lightsail propulsion](https://pubs.acs.org/doi/10.1021/acsphotonics.0c00768), _ACS Photonics_ **7**, 9, 2350-2355 (2020).
    