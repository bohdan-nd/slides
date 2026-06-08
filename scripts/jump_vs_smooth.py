"""
Continuous (smooth) vs discrete (teleport) — stacked vertically.

Top    — smooth ODE trajectory X_t vs t, same style as ode_trajectory.gif:
         dot{X} = 4 sin(2 pi t) - 0.4 X, X_0 = 0.5,
         faded full trajectory + bright growing trail + circular head.
Bottom — discrete state space with exactly two positions on the x-axis:
         "token" (left) and "[MASK]" (right). Each of N coloured tokens
         starts at "token" and, at its independent random jump time
         tau_i drawn from P(tau > t) = 1 - t, instantaneously teleports
         to "[MASK]" and turns grey. Tokens flip at different times.

Output: public/figures/jump_vs_smooth.gif
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# palette — matches the existing gifs (_common.py)
BLUE   = "#1f77b4"
GREEN  = "#2ca02c"
YELLOW = "#f4c20d"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
PINK   = "#e377c2"
MASK_GRAY = "#bdbdbd"
TEXT      = "#1F1B1A"
TEXT_SOFT = "#6E6967"

plt.rcParams.update({
    "font.family": "Arial",
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

# ---------- top: ode_trajectory dynamics --------------------------------
T_FINAL  = 1.0
N_FRAMES = 160
X_LIM    = (-0.4, 2.4)
X0       = 0.5


def velocity(t: float, x: np.ndarray) -> np.ndarray:
    return 4.0 * np.sin(2.0 * np.pi * t) - 0.4 * np.asarray(x)


def integrate(x0: float, t_eval: np.ndarray) -> np.ndarray:
    sol = solve_ivp(
        lambda t, x: [velocity(t, x[0])],
        (t_eval[0], t_eval[-1]),
        [x0],
        t_eval=t_eval,
        rtol=1e-8, atol=1e-10,
    )
    return sol.y[0]


ts = np.linspace(0.0, T_FINAL, N_FRAMES)
xs = integrate(X0, ts)

# ---------- bottom: per-token jump times --------------------------------
rng = np.random.default_rng(11)
N_TOKENS     = 6
TOKEN_LABELS = ["A", "B", "C", "D", "E", "F"]
TOKEN_COLORS = [BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK]
# Linear schedule: P(tau > t) = 1 - t  =>  tau ~ Uniform(0, 1).
jump_times = rng.uniform(low=0.08, high=0.95, size=N_TOKENS)

X_TOKEN = 0.0
X_MASK  = 1.0

# ---------- figure & layout ---------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(9.5, 6.6),
    gridspec_kw={"height_ratios": [2.0, 2.4]},
    constrained_layout=True,
)

# ---------- top panel ---------------------------------------------------
ax_top.set_xlim(0.0, 1.0)
ax_top.set_ylim(*X_LIM)
ax_top.set_xlabel(r"$t$")
ax_top.set_ylabel(r"$X_t$")
ax_top.set_title("Continuous (smooth)", fontsize=11,
                 color=TEXT, pad=8, loc="left")
ax_top.axhline(0, color="gray", lw=0.4, alpha=0.5)
ax_top.grid(alpha=0.15)

ax_top.plot(ts, xs, color=BLUE, lw=1.0, alpha=0.18)
ax_top.plot(ts[0], xs[0], color=BLUE, marker="s",
            ms=8, mec="black", mew=0.7)

(trail,) = ax_top.plot([], [], color=BLUE, lw=2.6)
(head,)  = ax_top.plot([], [], color=BLUE, marker="o",
                       ms=11, mec="black", mew=0.8)
t_text = ax_top.text(
    0.02, 0.94, "", transform=ax_top.transAxes, fontsize=11,
    bbox={"boxstyle": "round", "facecolor": "white",
          "alpha": 0.9, "edgecolor": "gray"},
)

# ---------- bottom panel: 2-state teleport ------------------------------
Y_SPACING = 1.6  # vertical gap between tokens
y_positions = [(N_TOKENS - 1 - i) * Y_SPACING for i in range(N_TOKENS)]

ax_bot.set_xlim(-0.25, 1.25)
ax_bot.set_ylim(-Y_SPACING * 0.6, (N_TOKENS - 1) * Y_SPACING + Y_SPACING * 0.6)
ax_bot.set_xticks([X_TOKEN, X_MASK])
ax_bot.set_xticklabels(["token", "[MASK]"], fontsize=11)
ax_bot.set_yticks([])
ax_bot.set_title("Discrete (jumps)", fontsize=11,
                 color=TEXT, pad=8, loc="left")
ax_bot.grid(False)
for spine in ("top", "right", "left"):
    ax_bot.spines[spine].set_visible(False)
ax_bot.spines["bottom"].set_color(TEXT_SOFT)

# destination marker (faint grey hollow circle on the right) for each row
for y in y_positions:
    ax_bot.plot(X_MASK, y, "o",
                markerfacecolor="white", markeredgecolor=TEXT_SOFT,
                markeredgewidth=1.0, markersize=14, alpha=0.4, zorder=2)

# the live token dots — one per token
dots       = []
dot_labels = []
for i, y in enumerate(y_positions):
    dot, = ax_bot.plot(
        [X_TOKEN], [y], "o",
        color=TOKEN_COLORS[i], markersize=20,
        markeredgecolor=TEXT, markeredgewidth=1.2, zorder=4,
    )
    lbl = ax_bot.text(
        X_TOKEN, y, TOKEN_LABELS[i],
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=5,
    )
    dots.append(dot)
    dot_labels.append(lbl)


def init():
    trail.set_data([], [])
    head.set_data([], [])
    t_text.set_text("")
    for i, (dot, lbl) in enumerate(zip(dots, dot_labels)):
        y = y_positions[i]
        dot.set_data([X_TOKEN], [y])
        dot.set_color(TOKEN_COLORS[i])
        lbl.set_position((X_TOKEN, y))
        lbl.set_text(TOKEN_LABELS[i])
        lbl.set_color("white")
    return ()


def update(frame: int):
    t_now = ts[frame]
    # top
    trail.set_data(ts[: frame + 1], xs[: frame + 1])
    head.set_data([ts[frame]], [xs[frame]])
    t_text.set_text(f"$t = {t_now:.2f}$")

    # bottom: per token, teleport to MASK once t crosses tau_i
    for i, (dot, lbl) in enumerate(zip(dots, dot_labels)):
        y = y_positions[i]
        if t_now >= jump_times[i]:
            dot.set_data([X_MASK], [y])
            dot.set_color(MASK_GRAY)
            lbl.set_position((X_MASK, y))
            lbl.set_text("")  # state is encoded by position
        else:
            dot.set_data([X_TOKEN], [y])
            dot.set_color(TOKEN_COLORS[i])
            lbl.set_position((X_TOKEN, y))
            lbl.set_text(TOKEN_LABELS[i])
            lbl.set_color("white")
    return ()


anim = FuncAnimation(
    fig, update, frames=N_FRAMES, init_func=init,
    interval=33, blit=False, repeat=True,
)

out_dir = Path(__file__).resolve().parents[1] / "public" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "jump_vs_smooth.gif"
anim.save(gif_path, writer=PillowWriter(fps=24))
print(f"Wrote {gif_path}")
