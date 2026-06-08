"""
Visualize the probability simplex of 3 tokens as a 2D plane in R^3.

For a vocabulary {A, B, C}, the simplex Delta^2 = {(p_A, p_B, p_C) : p_i >= 0,
sum p_i = 1} is a triangle with vertices at the one-hot vectors
(1,0,0), (0,1,0), (0,0,1). Continuous diffusion / flow matching over discrete
data with a one-hot representation operates on this 2D triangle inside R^3.

Renders a single static projection (PNG) showing the triangle as a tilted
2D plane embedded in 3D, with labelled vertices and a uniform-distribution
centroid. Palette matches the existing project gifs (flow.gif etc).

Output: public/figures/simplex_3tokens.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# palette — matches the existing gifs (_common.py in the lecture project)
BLUE   = "#1f77b4"
GREEN  = "#2ca02c"
YELLOW = "#f4c20d"
GRAY   = "#7f7f7f"
TEXT   = "#1F1B1A"

plt.rcParams.update({"font.family": "Arial"})

# one-hot vertices of the simplex (the three tokens)
A = np.array([1.0, 0.0, 0.0])  # token A
B = np.array([0.0, 1.0, 0.0])  # token B
C = np.array([0.0, 0.0, 1.0])  # token C
CENTROID = (A + B + C) / 3.0   # uniform distribution

fig = plt.figure(figsize=(6.5, 6.0))
ax  = fig.add_subplot(111, projection="3d")

# bounding box
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_zlim(0, 1.05)
ax.set_box_aspect((1, 1, 1))

# coordinate axes from origin
arrow_kw = {"color": GRAY, "lw": 1.2, "alpha": 0.6}
ax.plot([0, 1.05], [0, 0], [0, 0], **arrow_kw)
ax.plot([0, 0], [0, 1.05], [0, 0], **arrow_kw)
ax.plot([0, 0], [0, 0], [0, 1.05], **arrow_kw)
ax.text(1.10, 0, 0, r"$p_A$", fontsize=11, color=GRAY)
ax.text(0, 1.10, 0, r"$p_B$", fontsize=11, color=GRAY)
ax.text(0, 0, 1.12, r"$p_C$", fontsize=11, color=GRAY)

# the simplex triangle: filled translucent face + emphasised edges
triangle = Poly3DCollection(
    [[A, B, C]], alpha=0.18,
    facecolor=BLUE, edgecolor="none",
)
ax.add_collection3d(triangle)
for v1, v2 in [(A, B), (B, C), (C, A)]:
    ax.plot(
        [v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
        color=BLUE, lw=2.0, alpha=0.85,
    )

# vertices: coloured dots + labels
vertex_colors = [BLUE, GREEN, YELLOW]
vertex_labels = ["A", "B", "C"]
for v, col, label in zip([A, B, C], vertex_colors, vertex_labels):
    ax.scatter(*v, color=col, s=140,
               edgecolor=TEXT, linewidth=1.0, zorder=10)
    ax.text(v[0] + 0.04, v[1] + 0.04, v[2] + 0.06,
            label, fontsize=15, fontweight="bold", color=col)

# uniform-distribution centroid
ax.scatter(*CENTROID, color=TEXT, marker="*", s=160,
           edgecolor="white", linewidth=0.8, zorder=11)
ax.text(CENTROID[0] + 0.04, CENTROID[1] + 0.04, CENTROID[2] + 0.04,
        "uniform", fontsize=9, color=TEXT)

# cosmetics
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_zticks([0, 1])
ax.tick_params(labelsize=8, colors=GRAY)
ax.grid(False)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor("white")
    pane.pane.set_edgecolor(GRAY)
    pane.pane.set_alpha(0.05)

# fixed camera angle — lower elevation + off-diagonal azimuth so the
# triangle reads as a tilted plane in 3D (not a face-on flat triangle).
ax.view_init(elev=12, azim=22)

out_dir = Path(__file__).resolve().parents[1] / "public" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / "simplex_3tokens.png"
fig.savefig(png_path, dpi=180, bbox_inches="tight")
print(f"Wrote {png_path}")
