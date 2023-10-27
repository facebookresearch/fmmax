---
id: scattering.ScatteringMatrix
---

    ### `Class scattering.ScatteringMatrix():`
Stores the scattering matrix for a stack of layers.

The first layer in a stack is the "start" layer, and the last layer in the
stack is the "end" layer.

The scattering matrix relates the forward-going and backward-going waves
on the two sides of a layer stack, which are labeled `a` and `b` respectively.

Note that forward going fields are defined at the *start* of a layer while
backward-going fields are defined at the *end* of a layer, as depicted below.
This is discussed near equation 4.1 in [1999 Whittaker].

            |             |           |         |           |
            |   layer 0   |  layer 1  |   ...   |  layer N  |
            | start layer |           |         | end layer |
            |             |           |         |           |
             -> a_0                              -> a_N
                    b_0 <-                            b_N <-

Following the convention of [1999 Whittaker], the terms a_N and b_0 are
obtained from,

                a_N = s11 @ a_0 + s12 @ b_N
                b_0 = s21 @ a_0 + s22 @ b_N

Besides the actual scattering matrix element, the `ScatteringMatrix` stores
information about the start and end layers, which are needed to extend the
scattering matrix to include more layers.

#### Args:
- **s11** (None): Relates forward-going fields at start to forward-going fields at end.
- **s12** (None): Relates backward-going fields at end to forward-going fields at end.
- **s21** (None): Relates forward-going fields at start to backward-going fields at start.
- **s22** (None): Relates backward-going fields at end to backward-going fields at start.
- **start_layer_solve_result** (None): The eigensolve result for the start layer.
- **start_layer_thickness** (None): The start layer thickness.
- **end_layer_solve_result** (None): The eigensolve result for the end layer.
- **end_layer_thickness** (None): The end layer thickness.

