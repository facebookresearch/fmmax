"""Functions related to scattering matrix computation for the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

from fmmax import fmm, utils


@dataclasses.dataclass
class ScatteringMatrix:
    """Stores the scattering matrix for a stack of layers.

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

    Attributes:
        s11: Relates forward-going fields at start to forward-going fields at end.
        s12: Relates backward-going fields at end to forward-going fields at end.
        s21: Relates forward-going fields at start to backward-going fields at start.
        s22: Relates backward-going fields at end to backward-going fields at start.
        start_layer_solve_result: The eigensolve result for the start layer.
        start_layer_thickness: The start layer thickness.
        end_layer_solve_result: The eigensolve result for the end layer.
        end_layer_thickness: The end layer thickness.
    """

    s11: jnp.ndarray
    s12: jnp.ndarray
    s21: jnp.ndarray
    s22: jnp.ndarray

    start_layer_solve_result: fmm.LayerSolveResult
    start_layer_thickness: jnp.ndarray

    end_layer_solve_result: fmm.LayerSolveResult
    end_layer_thickness: jnp.ndarray


def stack_s_matrix(
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
) -> ScatteringMatrix:
    """Computes the s-matrix for a stack of layers.

    If only a single layer is provided, the scattering matrix is just the
    identity matrix, and start and end layer data is for the same layer.

    Args:
        layer_solve_results: The eigensolve results for layers in the stack.
        layer_thicknesses: The thicknesses for layers in the stack.

    Returns:
        The `ScatteringMatrix`.
    """
    return _stack_s_matrices(layer_solve_results, layer_thicknesses)[-1]


def stack_s_matrices_interior(
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
) -> Tuple[Tuple[ScatteringMatrix, ScatteringMatrix], ...]:
    """Computes scattering matrices before and after each layer in the stack.

    Specifically, for each layer `i` two `ScatteringMatrix` are returned. The
    first relates fields in the substack `0, ..., i`, while the second relates
    the fields in the substack `i, ..., N`, where `N` is the maximum layer
    index. These two scattering matrices are denoted as the corresponding
    to the "before" substack and the "after" substack.

    Args:
        layer_solve_results: The eigensolve results for layers in the stack.
        layer_thicknesses: The thicknesses for layers in the stack.

    Returns:
        The tuple of `(scattering_matrix_before, scattering_matrix_after)`.
    """
    before = _stack_s_matrices(layer_solve_results, layer_thicknesses)

    # Compute the scattering matrix for the substack "after" each layer. We do
    # this by computing the scattering matrix for the substack "before" each layer
    # in the reversed stack. Then, reverse each resulting scattering matrix.
    reverse = _stack_s_matrices(layer_solve_results[::-1], layer_thicknesses[::-1])
    after = tuple(
        [
            ScatteringMatrix(
                s11=s.s22,
                s12=s.s21,
                s21=s.s12,
                s22=s.s11,
                start_layer_solve_result=s.end_layer_solve_result,
                start_layer_thickness=s.end_layer_thickness,
                end_layer_solve_result=s.start_layer_solve_result,
                end_layer_thickness=s.start_layer_thickness,
            )
            for s in reverse[::-1]
        ]
    )
    return tuple(zip(before, after))


def _stack_s_matrices(
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
) -> Tuple[ScatteringMatrix, ...]:
    """Computes the s-matrices for a stack of layers.

    The matrices are for various substacks: the first scattering matrix is
    for the first layer only, the second is for the stack consisting of the
    first and second layer, etc.

    Args:
        layer_solve_results: The eigensolve results for layers in the stack.
        layer_thicknesses: The thicknesses for layers in the stack.

    Returns:
        The tuple of `ScatteringMatrix`.
    """
    if len(layer_solve_results) != len(layer_thicknesses):
        raise ValueError(
            f"`layer_solve_results` and `layer_thicknesses` should have the same "
            f"length but got {len(layer_solve_results)} and {len(layer_thicknesses)}."
        )

    # The initial scattering matrix is just the identity matrix, with the
    # necessary batch dimensions.
    eye = utils.diag(jnp.ones_like(layer_solve_results[0].eigenvalues))
    s_matrix = ScatteringMatrix(
        s11=eye,
        s12=jnp.zeros_like(eye),
        s21=jnp.zeros_like(eye),
        s22=eye,
        start_layer_solve_result=layer_solve_results[0],
        start_layer_thickness=layer_thicknesses[0],
        end_layer_solve_result=layer_solve_results[0],
        end_layer_thickness=layer_thicknesses[0],
    )

    # TODO(mfschubert): Figure out how to use `jax.lax.fori_loop` or similar.
    s_matrices = [s_matrix]
    for layer_solve_result, layer_thickness in zip(
        layer_solve_results[1:], layer_thicknesses[1:]
    ):
        s_matrices.append(
            append_layer(s_matrices[-1], layer_solve_result, layer_thickness)
        )
    return tuple(s_matrices)


def stack_s_matrix_scan(
    layer_solve_results: fmm.LayerSolveResult,
    layer_thicknesses: jnp.ndarray,
) -> ScatteringMatrix:
    """Computes the stack matrix for a stack of layers.

    Unlike `stack_s_matrix`, this function uses a scan operation rather than a python
    for loop, which can lead to significantly smaller programs and shorter compile times.
    However, it requires that the layer solve results and thicknesses for be represented
    in the leading batch dimension of the arguments.

    This function is best used when the eigensolve for each layer is also done in a
    batched manner.

    Args:
        layer_solve_results: The layer solve results for all layers in the stack.
        layer_thicknesses: The layer thicknesses for all layers in the stack.

    Returns:
        The scattering matrix for the stack.
    """
    assert layer_thicknesses.ndim == 1
    if layer_solve_results.batch_shape[0] != len(layer_thicknesses):
        raise ValueError(
            f"`layer_solve_results` and `layer_thicknesses` should be compatible (i.e. "
            f"correspond to the same number of layers) but inferred layer numbers of "
            f"{layer_solve_results.batch_shape[0]} and {layer_thicknesses.shape[0]}."
        )

    eye = utils.diag(jnp.ones(layer_solve_results.eigenvalues.shape[1:], dtype=complex))
    start_solve_result = tree_util.tree_map(lambda x: x[0, ...], layer_solve_results)
    s_matrix = ScatteringMatrix(
        s11=eye,
        s12=jnp.zeros_like(eye),
        s21=jnp.zeros_like(eye),
        s22=eye,
        start_layer_solve_result=start_solve_result,
        start_layer_thickness=layer_thicknesses[0],
        end_layer_solve_result=start_solve_result,
        end_layer_thickness=layer_thicknesses[0],
    )

    def scan_fn(s_matrix, x):
        next_layer_solve_result, next_layer_thickness = x
        s_matrix = append_layer(s_matrix, next_layer_solve_result, next_layer_thickness)
        return s_matrix, s_matrix

    s_matrix, _ = jax.lax.scan(
        scan_fn,
        init=s_matrix,
        xs=(
            tree_util.tree_map(lambda x: x[1:], layer_solve_results),
            layer_thicknesses[1:],
        ),
    )
    return s_matrix


def redheffer_star_product(
    a: ScatteringMatrix,
    b: ScatteringMatrix,
) -> ScatteringMatrix:
    """Compute the Redheffer star product of two scattering matrices."""
    a_extended = append_layer(a, b.start_layer_solve_result, b.start_layer_thickness)
    a11, a12, a21, a22 = a_extended.s11, a_extended.s12, a_extended.s21, a_extended.s22
    b11, b12, b21, b22 = b.s11, b.s12, b.s21, b.s22

    # See https://en.wikipedia.org/wiki/Redheffer_star_product
    eye = utils.diag(jnp.ones_like(a11[..., 0]))
    s11 = b11 @ jnp.linalg.solve(eye - a12 @ b21, a11)
    s12 = b12 + b11 @ jnp.linalg.solve(eye - a12 @ b21, a12 @ b22)
    s21 = a21 + a22 @ jnp.linalg.solve(eye - b21 @ a12, b21 @ a11)
    s22 = a22 @ jnp.linalg.solve(eye - b21 @ a12, b22)
    return ScatteringMatrix(
        s11=s11,
        s12=s12,
        s21=s21,
        s22=s22,
        start_layer_solve_result=a.start_layer_solve_result,
        start_layer_thickness=a.start_layer_thickness,
        end_layer_solve_result=b.end_layer_solve_result,
        end_layer_thickness=b.end_layer_thickness,
    )


def append_layer(
    s_matrix: ScatteringMatrix,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
) -> ScatteringMatrix:
    """Returns new scattering matrix for the stack with an appended layer.

    Args:
        s_matrix: The existing scattering matrix.
        next_layer_solve_result: The eigensolve result for the layer to append.
        next_layer_thickness: The thickness for the layer to append.

    Returns:
        The new `ScatteringMatrix`.
    """
    s11_next, s12_next, s21_next, s22_next = _extend_s_matrix(
        s_matrix_blocks=(s_matrix.s11, s_matrix.s12, s_matrix.s21, s_matrix.s22),
        layer_solve_result=s_matrix.end_layer_solve_result,
        layer_thickness=s_matrix.end_layer_thickness,
        next_layer_solve_result=next_layer_solve_result,
        next_layer_thickness=next_layer_thickness,
    )
    return ScatteringMatrix(
        s11=s11_next,
        s12=s12_next,
        s21=s21_next,
        s22=s22_next,
        start_layer_solve_result=s_matrix.start_layer_solve_result,
        start_layer_thickness=s_matrix.start_layer_thickness,
        end_layer_solve_result=next_layer_solve_result,
        end_layer_thickness=next_layer_thickness,
    )


def prepend_layer(
    s_matrix: ScatteringMatrix,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
) -> ScatteringMatrix:
    """Returns new scattering matrix for the stack with a prepended layer.

    Args:
        s_matrix: The existing scattering matrix.
        next_layer_solve_result: The eigensolve result for the layer to append.
        next_layer_thickness: The thickness for the layer to append.

    Returns:
        The new `ScatteringMatrix`.
    """
    # To prepend a layer, we compute the scattering matrix that results from
    # appending the layer to the reversed stack. The scattering matrix for
    # the reversed stack is simply the re-ordering of the matrix blocks, i.e.
    # s11 become s22, s12 becomes s21, etc.
    s22_next, s21_next, s12_next, s11_next = _extend_s_matrix(
        s_matrix_blocks=(s_matrix.s22, s_matrix.s21, s_matrix.s12, s_matrix.s11),
        layer_solve_result=s_matrix.start_layer_solve_result,
        layer_thickness=s_matrix.start_layer_thickness,
        next_layer_solve_result=next_layer_solve_result,
        next_layer_thickness=next_layer_thickness,
    )
    return ScatteringMatrix(
        s11=s11_next,
        s12=s12_next,
        s21=s21_next,
        s22=s22_next,
        start_layer_solve_result=next_layer_solve_result,
        start_layer_thickness=next_layer_thickness,
        end_layer_solve_result=s_matrix.end_layer_solve_result,
        end_layer_thickness=s_matrix.end_layer_thickness,
    )


def _extend_s_matrix(
    s_matrix_blocks: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extends the scattering matrix, adding a layer to the end.

    The approach here follows section 5 of [1999 Whittaker].

    Args:
        s_matrix_blocks: The elements `(s11, s12, s21, s22)`.
        layer_solve_result: The eigensolve result of the ending layer.
        layer_thickness: The thickness of the ending layer.
        next_layer_solve_result: The eigensolve result for the layer to append.
        next_layer_thickness: The thickness for the layer to append.

    Returns:
        The new `ScatteringMatrix`.
    """
    # Alias for brevity: eigenvalues, eigenvectors, and omega-k matrix.
    q = layer_solve_result.eigenvalues
    phi = layer_solve_result.eigenvectors
    omega_k = layer_solve_result.omega_script_k_matrix

    next_q = next_layer_solve_result.eigenvalues
    next_phi = next_layer_solve_result.eigenvectors
    next_omega_k = next_layer_solve_result.omega_script_k_matrix

    # Compute the interface matrices following equation 5.3 of [1999 Whittaker].
    # These make use the matrix form of the orthogonality relation of equation 3.9
    # in [1999 Whittaker], i.e. `phi_T @ omega_k @ phi = 1`, to compute `phi_T`.
    # Throughout, we use optimized expressions which avoid matrix inversion and
    # matrix-matrix multiplications. More straightforward implementations are in
    # comments.
    #
    # phi_T = jnp.linalg.inv(omega_k @ phi)
    # term1 = diag(q) @ phi_T @ next_omega_k @ next_phi @ diag(1 / next_q)
    # term1 = q[..., :, jnp.newaxis] * jnp.linalg.solve(
    #     omega_k @ phi,
    #     next_omega_k @ next_phi * (1 / next_q)[..., jnp.newaxis, :],
    # )
    term1 = utils.diag(q) @ jnp.linalg.solve(
        omega_k @ phi,
        next_omega_k @ next_phi @ utils.diag(1 / next_q),
    )
    # term2 = phi_T @ omega_k @ next_phi
    term2 = jnp.linalg.solve(omega_k @ phi, omega_k @ next_phi)
    i11 = i22 = 0.5 * (term1 + term2)
    i12 = i21 = 0.5 * (-term1 + term2)

    # Phase terms \hat{f}(d) defined near equation 4.2 of [1999 Whittaker]. These
    # describe phase accumulated by propagating across a layer for each eigenmode.
    fd = utils.diag(jnp.exp(1j * q * layer_thickness))
    fd_next = utils.diag(jnp.exp(1j * next_q * next_layer_thickness))

    # Update the s-matrix to include the present layer, following the recipe
    # given in equation 5.4 of [1999 Whittaker].
    s11, s12, s21, s22 = s_matrix_blocks

    # s11_next = inv(i11 - diag(fd) @ s12 @ i21) @ diag(fd) @ s11
    term3 = i11 - fd @ s12 @ i21
    s11_next = jnp.linalg.solve(term3, fd @ s11)
    # s12_next = inv(i11 - diag(fd) @ s12 @ i21) @ (diag(fd) @ s12 @ i22 - i12) @ diag(fd_next)
    s12_next = jnp.linalg.solve(term3, (fd @ s12 @ i22 - i12) @ fd_next)
    s21_next = s22 @ i21 @ s11_next + s21
    # s22_next = s22 @ i21 @ s12_next + s22 @ i22 @ diag(fd_next)
    s22_next = s22 @ i21 @ s12_next + s22 @ i22 @ fd_next

    return (s11_next, s12_next, s21_next, s22_next)


def set_end_layer_thickness(
    s_matrix: ScatteringMatrix,
    thickness: jnp.ndarray,
) -> ScatteringMatrix:
    """Returns a new `ScatteringMatrix` with a modified end layer thickness.

    Args:
        s_matrix: The initial `ScatteringMatrix`.
        thickness: The desired thickness of the layer.

    Returns:
        The new `ScatteringMatrix`.
    """
    q = s_matrix.end_layer_solve_result.eigenvalues
    fd = jnp.exp(1j * q * (thickness - s_matrix.end_layer_thickness))
    return ScatteringMatrix(
        s11=s_matrix.s11,
        s12=s_matrix.s12 * fd[..., jnp.newaxis, :],
        s21=s_matrix.s21,
        s22=s_matrix.s22 * fd[..., jnp.newaxis, :],
        start_layer_solve_result=s_matrix.start_layer_solve_result,
        start_layer_thickness=s_matrix.start_layer_thickness,
        end_layer_solve_result=s_matrix.end_layer_solve_result,
        end_layer_thickness=thickness,
    )


def set_start_layer_thickness(
    s_matrix: ScatteringMatrix,
    thickness: jnp.ndarray,
) -> ScatteringMatrix:
    """Returns a new `ScatteringMatrix` with a modified start layer thickness.

    Args:
        s_matrix: The initial `ScatteringMatrix`.
        thickness: The desired thickness of the layer.

    Returns:
        The new `ScatteringMatrix`.
    """
    q = s_matrix.start_layer_solve_result.eigenvalues
    fd = jnp.exp(1j * q * (thickness - s_matrix.start_layer_thickness))
    return ScatteringMatrix(
        s11=s_matrix.s11 * fd[..., jnp.newaxis, :],
        s12=s_matrix.s12,
        s21=s_matrix.s21 * fd[..., jnp.newaxis, :],
        s22=s_matrix.s22,
        start_layer_solve_result=s_matrix.start_layer_solve_result,
        start_layer_thickness=thickness,
        end_layer_solve_result=s_matrix.end_layer_solve_result,
        end_layer_thickness=s_matrix.end_layer_thickness,
    )


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


jax.tree_util.register_pytree_node(
    ScatteringMatrix,
    lambda x: (
        (
            x.s11,
            x.s12,
            x.s21,
            x.s22,
            x.start_layer_solve_result,
            x.start_layer_thickness,
            x.end_layer_solve_result,
            x.end_layer_thickness,
        ),
        None,
    ),
    lambda _, x: ScatteringMatrix(*x),
)
