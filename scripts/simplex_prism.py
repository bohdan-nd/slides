"""
Joint state space R^d x Delta^(|V|-1) shown as an *unbounded* triangular
prism: only the three lateral simplex-shaped sides really exist; the
spatial axis itself is unbounded. So there are no closed "end faces",
just three lateral edges (with chevrons hinting at extension) and a
couple of faint dashed simplex cross-sections to make the slice
structure visible.

A separate toy molecule is drawn *above* the prism (well off the
cross-section plane) so it cannot be misread as sitting on a parallel
simplex slice.

  spatial axis  -->  R^d   (unbounded, along the prism length)
  simplex       -->  Delta^2 (every perpendicular cross-section)

Output: public/figures/simplex_prism.gif
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# simplex / prism palette (same as flow.gif etc.)
BLUE   = "#1f77b4"
GREEN  = "#2ca02c"
YELLOW = "#f4c20d"
GRAY   = "#7f7f7f"
TEXT   = "#1F1B1A"


plt.rcParams.update({"font.family": "Arial"})

# Equilateral simplex in the (y, z) plane
A_2d = np.array([-0.5, 0.0])
B_2d = np.array([ 0.5, 0.0])
C_2d = np.array([ 0.0, np.sqrt(3) / 2])

def lift(p2: np.ndarray, x: float) -> np.ndarray:
    return np.array([x, p2[0], p2[1]])

# extent shown on the spatial axis (it is unbounded; this is just the crop)
X_LEFT  = -1.6
X_RIGHT =  1.8
CROSS_SECTIONS = [-1.0, -0.3, 0.4, 1.1]   # dashed simplex slices

def simplex_face(x: float):
    return [lift(A_2d, x), lift(B_2d, x), lift(C_2d, x)]


# ---------- one molecule — a single point INSIDE the joint state space --
# An entire molecule is represented as ONE point inside the prism: its
# x-coordinate is the spatial position (1D in this collapsed picture),
# its (y, z) is the position inside the simplex cross-section (its token
# distribution).

# barycentric weights over (A, B, C) for the molecule's token distribution
_w = np.array([0.30, 0.40, 0.30])
_mol_simplex_2d = _w[0] * A_2d + _w[1] * B_2d + _w[2] * C_2d

MOL_X      = 1.40   # well past the labelled simplex slice at x = 0.4
mol_point  = np.array([MOL_X, _mol_simplex_2d[0], _mol_simplex_2d[1]])
MOL_COLOR  = "#d62728"     # red — distinct from simplex palette


# ---------- animation ---------------------------------------------------
N_FRAMES = 90
FPS      = 6

fig = plt.figure(figsize=(7.5, 4.4))
ax  = fig.add_subplot(111, projection="3d")
# push the 3D axes to fill the figure — kills most of the default whitespace
fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)


def draw_lateral_edge(v_2d: np.ndarray):
    """Three lateral edges of the prism — fade slightly at the ends, with
    a small chevron suggesting extension to +/- infinity."""
    pL, pR = lift(v_2d, X_LEFT), lift(v_2d, X_RIGHT)
    ax.plot([pL[0], pR[0]], [pL[1], pR[1]], [pL[2], pR[2]],
            color=BLUE, lw=2.0, alpha=0.85)
    # chevrons at both ends to telegraph extension
    chev = 0.18
    for px, sign in [(pL, -1), (pR, +1)]:
        ax.plot([px[0], px[0] + sign * chev],
                [px[1], px[1]], [px[2], px[2]],
                color=BLUE, lw=1.2, alpha=0.55, ls="--")


def draw_cross_section(x: float, prominent: bool = False):
    """A simplex slice at given spatial coord, drawn as a dashed triangle."""
    f = simplex_face(x)
    closed = f + [f[0]]
    xs = [p[0] for p in closed]
    ys = [p[1] for p in closed]
    zs = [p[2] for p in closed]
    if prominent:
        ax.plot(xs, ys, zs, color=BLUE, lw=1.6, alpha=0.85)
    else:
        ax.plot(xs, ys, zs, color=BLUE, lw=0.9, alpha=0.40, ls=":")


def draw_scene():
    ax.clear()

    # three lateral edges (the only "real" boundary)
    for v_2d in (A_2d, B_2d, C_2d):
        draw_lateral_edge(v_2d)

    # a few dashed simplex cross-sections, one prominent
    for k, x_cs in enumerate(CROSS_SECTIONS):
        draw_cross_section(x_cs, prominent=(k == 2))

    # token vertex labels at the prominent cross-section
    x_label = CROSS_SECTIONS[2]
    for v_2d, col, label in zip(
        [A_2d, B_2d, C_2d], [BLUE, GREEN, YELLOW], ["A", "B", "C"],
    ):
        p = lift(v_2d, x_label)
        ax.scatter(*p, color=col, s=120,
                   edgecolor=TEXT, linewidth=1.0, zorder=10)
        ax.text(p[0] + 0.05, p[1] + 0.06, p[2] + 0.08,
                label, fontsize=12, fontweight="bold", color=col)

    # spatial axis arrow below the prism (with chevrons at both ends)
    z_axis = -0.22
    ax.plot([X_LEFT - 0.35, X_RIGHT + 0.35], [0, 0], [z_axis, z_axis],
            color=GRAY, lw=1.4, alpha=0.75)
    for sign in (-1, +1):
        x_end = (X_LEFT - 0.35) if sign < 0 else (X_RIGHT + 0.35)
        ax.plot([x_end, x_end + sign * 0.12], [0, 0],
                [z_axis, z_axis], color=GRAY, lw=1.4, alpha=0.55, ls="--")

    # simplex label
    ax.text(X_LEFT - 0.5, 0.05, 0.95,
            r"$\Delta^{|V|-1}$",
            fontsize=12, color=BLUE, fontweight="bold")

    # --- the molecule: one point in the joint state space --------------
    # dashed drop-line down to the spatial axis (showing its x)
    z_axis = -0.22
    ax.plot([mol_point[0], mol_point[0]],
            [mol_point[1], mol_point[1]],
            [mol_point[2], z_axis],
            color=MOL_COLOR, lw=1.2, ls=":", alpha=0.7, zorder=6)
    ax.scatter(mol_point[0], mol_point[1], z_axis,
               color=MOL_COLOR, s=40, alpha=0.85, zorder=6)

    # the molecule itself
    ax.scatter(*mol_point, color=MOL_COLOR, s=320,
               edgecolor=TEXT, linewidth=1.2, zorder=12)

    # cosmetics
    ax.set_xlim(X_LEFT - 0.5, X_RIGHT + 0.5)
    ax.set_ylim(-0.85, 0.85)
    ax.set_zlim(-0.45, 1.05)
    ax.set_box_aspect((X_RIGHT - X_LEFT + 1.0, 1.7, 1.4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(GRAY)
        axis.pane.set_alpha(0.04)
        axis.line.set_color(GRAY)


def update(frame: int):
    draw_scene()
    angle = -65 + 360 * frame / N_FRAMES
    ax.view_init(elev=14, azim=angle)
    return []


anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

out_dir = Path(__file__).resolve().parents[1] / "public" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "simplex_prism.gif"
anim.save(gif_path, writer=PillowWriter(fps=FPS))
print(f"Wrote {gif_path}")
