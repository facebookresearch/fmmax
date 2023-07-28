# FMMAX: Fourier Modal Method with Jax

FMMAX is a an implementation of the Fourier modal method (FMM) in [JAX](https://github.com/google/jax). 

The FMM -- also known as rigorous coupled wave analysis (RCWA) -- is a semianalytical method that solves Maxwell's equations in periodic stratified media, where in-plane directions are treated with a truncated Fourier basis and the normal direction is handled by a scattering matrix approach [1999 Whittaker, 2012 Liu, 2020 Jin]. This allows certain classes of structures to be modeled with relatively low computational cost.

Our use of JAX enables GPU acceleration and automatic differentiation of FMM simulations. Besides these features, FMMAX is differentiated from other codes by its support for Brillouin zone integration and advanced vector FMM formulations which improve convergence.

## Brillouin zone integration
Brillouin zone integration [2022 Lopez-Fraguas] allows modeling of localized sources in periodic structures. Check out the `crystal` example to see how we model a Gaussian beam incident upon a photonic crystal slab, or to model an isolated dipole embedded within the slab. The Gaussian beam fields are shown below.

![Gaussian beam incident on photonic crystal](/img/crystal_beam.gif)

## Vector FMM formulations
Vector FMM formulations introduce local coordinate systems at each point in the unit cell, which are normal and tangent to all interfaces. This allows normal and tangent field components to be treated differently and improves convergence. FMMAX implements several vector formulations of the FMM, with automatic vector field generation based on functional minimization similar to [2012 Liu]. We implement the _Pol_, _Normal_, and _Jones_ methods of that reference, and introduce a new _Jones direct_ method which we have found to have superior convergence. The `vector_fields` example computes vector fields by these methods for an example structure.

![Comparison of automatically-generated vector fields](/img/vector_fields.png)

## FMM Conventions
- The speed of light, vacuum permittivity, and vacuum permeability are all 1.
- Fields evolve in time as $\exp(-i \omega t)$.
- If $\mathbf{u}$ and $\mathbf{v}$ are the primitive lattice vectors, the unit cell is defined by the parallelogram with vertices at $\mathbf{0}$, $\mathbf{u}$, $\mathbf{u} + \mathbf{v}$, and $\mathbf{v}$.
- For quantities defined on a grid (such as the permittivity distribution of a patterned layer) the value at grid index (0, 0) corresponds to the value at physical location $\mathbf{0}$.
- The scattering matrix block $\mathbf{S}_{11}$ relates incident and transmitted forward-going fields, and other blocks have corresponding definitions. This differs from the convention e.g. in photonic integrated circuits.

## Batching
Batched calculations are supported, and should be used where possible to avoid looping. The batch axes are the leading axes, except for the wave amplitudes and electromagnetic fields, where a trailing batch axis is assumed. This allows e.g. computing the transmission through a structure for multiple polarizations via a matrix-matrix operation (`transmitted_amplitudes = S11 @ incident_amplitudes`), rather than a batched matrix-vector operation.

## Citing FMMAX

If you use FMMAX, please consider citing our paper,

```
@unpublished{schubert_fmm_2023,
  title = {Fourier modal method for inverse design of metasurface-enhanced micro-LEDs},
  author = {Schubert, Martin F. and Hammond, Alec},
  note = {Manuscript in preparation},
  year = {2023},
}
```

## License
FMMAX is licensed under the [MIT license](https://github.com/facebookresearch/fmmax/blob/main/LICENSE).

## References
- [2012 Liu] V. Liu and S. Fan, [S4: A free electromagnetic solver for layered structures structures](https://www.sciencedirect.com/science/article/pii/S0010465512001658), _Comput. Phys. Commun._ **183**, 2233-2244 (2012).

- [1999 Whittaker] D. M. Whittaker and I. S. Culshaw, [Scattering-matrix treatment of patterned multilayer photonic structures](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.60.2610), _Phys. Rev. B_ **60**, 2610 (1999).

- [2020 Jin] W. Jin, W. Li, M. Orenstein, and S. Fan [Inverse design of lightweight broadband reflector for relativistic lightsail propulsion](https://pubs.acs.org/doi/10.1021/acsphotonics.0c00768), _ACS Photonics_ **7**, 9, 2350-2355 (2020).

- [2022 Lopez-Fraguas] E. Lopez-Fraguas, F. Binkowski, S. Burger, B. Garcia-Camara, R. Vergaz, C. Becker and P. Manley [Tripling the light extraction efficiency of a deep ultraviolet LED using a nanostructured p-contact](https://www.nature.com/articles/s41598-022-15499-7), _Scientific Reports_ **12**, 11480 (2022).
    
