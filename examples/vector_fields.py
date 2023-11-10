"""Example which exercises automatic vector field generation.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt  # type: ignore[import]

from fmmax import basis, vector


def plot_vector_fields(
    dim: int = 100, interval: int = 10, savefig: bool = True
) -> None:
    """Generate a figure that compares different vector field schemes."""

    # Generate the pattern; a circular feature in a square unit cell.
    primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
    x, y = jnp.meshgrid(
        jnp.linspace(-0.5, 0.5, dim),
        jnp.linspace(-0.5, 0.5, dim),
        indexing="ij",
    )
    arr = (x**2 + y**2 < 0.4**2).astype(float)
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=200,
        truncation=basis.Truncation.CIRCULAR,
    )

    fig = plt.figure(figsize=(7, 4.5))

    for i, scheme_name in enumerate(vector.VECTOR_FIELD_SCHEMES):
        tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme_name](
            arr, expansion, primitive_lattice_vectors
        )

        ax = plt.subplot(2, 4, i + 1)
        _plot_vector_field(
            ax, arr, tx, ty, scale=0.7, interval=interval, title=scheme_name
        )

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05)

    if savefig:
        fig.savefig("vector_fields.png", bbox_inches="tight")


# -----------------------------------------------------------------------------
# Plotting functions.
# -----------------------------------------------------------------------------


def _plot_vector_field(
    ax: plt.Axes,
    arr: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    scale: float = 0.7,
    interval: int = 1,
    lw: float = 0.25,
    title: str = "",
) -> None:
    """Plots an overlay of `arr` with the vector field."""
    im = ax.imshow(arr, cmap="gray")
    im.set_clim((-5 * float(jnp.amax(arr) - jnp.amin(arr)), float(jnp.amax(arr))))

    x, y = jnp.meshgrid(*[jnp.arange(d) for d in vx.shape], indexing="ij")

    x = _sample(x, interval)
    y = _sample(y, interval)
    vx = _sample(vx, interval)
    vy = _sample(vy, interval)

    if jnp.any(jnp.iscomplex(vx)) or jnp.any(jnp.iscomplex(vy)):
        _plot_polarization_elipses(ax, x, y, vx, vy, scale=scale * interval, lw=lw)
    else:
        _plot_polarization_vectors(ax, x, y, vx, vy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def _sample(arr: jnp.ndarray, interval: int) -> jnp.ndarray:
    """Samples every elements from `arr`, selecting every `interval` element."""
    return arr[(interval // 2) :: interval, (interval // 2) :: interval]


def _plot_polarization_vectors(
    ax: plt.Axes, x: jnp.ndarray, y: jnp.ndarray, vx: jnp.ndarray, vy: jnp.ndarray
) -> None:
    """Plots a real vector field."""
    ax.quiver(y, x, -vy, vx)


def _plot_polarization_elipses(
    ax: plt.Axes,
    x: jnp.ndarray,
    y: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    scale: float,
    lw: float,
) -> None:
    """Plots a complex vector field."""
    t = jnp.concatenate([jnp.linspace(0, 2 * jnp.pi), jnp.asarray((jnp.nan,))])[
        jnp.newaxis, :
    ]
    x = x.flatten()[:, jnp.newaxis]
    y = y.flatten()[:, jnp.newaxis]
    vx = vx.flatten()[:, jnp.newaxis]
    vy = vy.flatten()[:, jnp.newaxis]
    xplot = x + jnp.real(vx * jnp.exp(1j * t)) * scale / 2
    yplot = y + jnp.real(vy * jnp.exp(1j * t)) * scale / 2
    ax.plot(yplot.flatten(), xplot.flatten(), lw=lw, color="k")


if __name__ == "__main__":
    plot_vector_fields()
