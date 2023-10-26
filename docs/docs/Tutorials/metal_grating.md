# Metal Grating

The full script can be found [here](https://github.com/facebookresearch/fmmax/blob/main/examples/metal_grating.py).


```python
resolution_nm: float = 1.0 # The rasterization resolution for patterned layers.
pitch_nm: float = 180.0 # The grating pitch, in nanometers.
grating_width_nm: float = 60.0 # The width of the lines comprising the grating.

x_nm, _ = jnp.meshgrid(
    jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
    jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
    indexing="ij",
)
density = (jnp.abs(x_nm) <= grating_width_nm / 2).astype(float)
```

```python
from fmmax import utils

permittivity_ambient: complex = 1.0 + 0.0j # The permittivity of the ambient.
permittivity_planarization: complex = 2.25 + 0.0j # The permittivity of media encapsulating grating.
permittivity_substrate: complex = -7.632 + 0.731j # The permittivity of the substrate below the grating

permittivities = [
    jnp.asarray([[permittivity_ambient]]),
    jnp.asarray([[permittivity_planarization]]),
    utils.interpolate_permittivity(
        permittivity_solid=jnp.asarray(permittivity_substrate),
        permittivity_void=jnp.asarray(permittivity_planarization),
        density=density,
    ),
    jnp.asarray([[permittivity_substrate]]),
]
```

```python
planarization_thickness_nm: float = 20.0
grating_thickness_nm: float = 80.0 # The height of the grating.

thicknesses = [0, planarization_thickness_nm, grating_thickness_nm, 0]
```

```python
in_plane_wavevector = jnp.asarray([0.0, 0.0])
```

```python
primitive_lattice_vectors = basis.LatticeVectors(
    u=jnp.asarray([pitch_nm, 0.0]), v=jnp.asarray([0.0, pitch_nm])
)
```

```python
from fmmax import basis

approximate_num_terms: int = 20
truncation: basis.Truncation = basis.Truncation.CIRCULAR

expansion = basis.generate_expansion(
    primitive_lattice_vectors=primitive_lattice_vectors,
    approximate_num_terms=approximate_num_terms,
    truncation=truncation,
)
```

```python
from fmmax import fmm

formulation: fmm.Formulation = fmm.Formulation.FFT
wavelength_nm: float = 500.0 # The excitation wavelength, in nanometers.

layer_solve_results = [
    fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(wavelength_nm),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=p,
        expansion=expansion,
        formulation=formulation,
    )
    for p in permittivities
]
```

```python
from fmmax import scattering

s_matrix = scattering.stack_s_matrix(
    layer_solve_results=layer_solve_results,
    layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
)
```

```python
r_te = s_matrix.s21[0, 0]
r_tm = s_matrix.s21[expansion.num_terms, expansion.num_terms]
```

```python

def convergence_study(
    approximate_num_terms: Tuple[int, ...] = NUM_TERMS_SWEEP,
    truncations: Tuple[basis.Truncation, ...] = (
        basis.Truncation.CIRCULAR,
        basis.Truncation.PARALLELOGRAMIC,
    ),
    fmm_formulations: Tuple[fmm.Formulation, ...] = (
        fmm.Formulation.FFT,
        fmm.Formulation.JONES_DIRECT,
        fmm.Formulation.JONES,
        fmm.Formulation.NORMAL,
        fmm.Formulation.POL,
    ),
) -> Tuple[Tuple[fmm.Formulation, basis.Truncation, int, complex, complex], ...]:
    """Sweeps over number of terms and fmm formulations to study convergence."""
    results = []
    for formulation, truncation, n in itertools.product(
        fmm_formulations,
        truncations,
        approximate_num_terms,
    ):
        num_terms, r_te, r_tm = simulate_grating(
            approximate_num_terms=n,
            truncation=truncation,
            formulation=formulation,
        )
        results.append((formulation, truncation, num_terms, r_te, r_tm))
        print(
            f"{formulation.value}/{truncation.value}/n={num_terms}: "
            f"r_te={complex(r_te):.3f}, r_tm={complex(r_tm):.3f}"
        )
    return tuple(results)
```
