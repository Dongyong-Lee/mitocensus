"""Numbered badges and key over the object-check render (light and dark variants).

Statements follow docs/BIOLOGICAL_ANALYSIS.md (object checks, section-support rule, split-label screen)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
THEMES = {"light": dict(title="#1F2328", body="#57606A", ring="#FFFFFF", lead="#57606A"),
          "dark": dict(title="#F0F6FC", body="#9198A1", ring="#0D1117", lead="#9198A1")}
KEY = [("1", "#1F5F99", "#FFFFFF", "QC-passing", "Inside the crop, in one piece and\nthrough enough sections: measured."),
       ("2", "#C3C9D0", "#1F2328", "Border-truncated", "Leaves the crop (faint part outside),\nso its size and shape are truncated."),
       ("3", "#7A838C", "#FFFFFF", "Disconnected", "One label on separated pieces."),
       ("4", "#CC79A7", "#1F2328", "Fewer than three sections", "Flat label in one or two sections; its\nminor axis is set by the voxel spacing,\nso it can look elongated."),
       ("5", "#CC79A7", "#1F2328", "Split across sections", "One object carried by two labels;\nflagged, never merged.")]
OFFSET = {"1": (-70, -10), "2": (20, -80), "3": (0, 105), "4": (95, 40), "5": (75, -10)}   # badge offset (render px)
W, H = 1400, 800

im = Image.open(SRC / "checks_render.png").convert("RGBA")
a = np.asarray(im)[..., 3]
ys, xs = np.nonzero(a > 12)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
pad = 20
crop = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, im.width), min(y1 + pad, im.height))
im = im.crop(crop)
scale = min(900 / im.width, 690 / im.height)
im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
arr = np.asarray(im)
anch = json.loads((SRC / "checks_anchors.json").read_text())["anchors_px"]
RX, RY = 10, 90


def to_canvas(p):
    return RX + (p[0] - crop[0]) * scale, RY + (p[1] - crop[1]) * scale


for theme, t in THEMES.items():
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ax.imshow(arr, extent=(RX, RX + arr.shape[1], RY + arr.shape[0], RY), interpolation="lanczos", zorder=1)
    ax.text(16, 22, "What the object checks see", fontsize=30, fontweight="bold", color=t["title"], va="top")
    ax.text(16, 62, "A crop of eight serial sections with synthetic shapes. Rings mark where an object meets a section.",
            fontsize=18.5, color=t["body"], va="top")
    for num, fc, tc, _, _ in KEY:
        targets = [anch[num]] if num != "3" else [anch["3a"], anch["3b"]]   # one label, two pieces: two leaders
        pts = [to_canvas(p) for p in targets]
        mx, my = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
        dx, dy = OFFSET[num]
        bx, by = mx + dx * scale, my + dy * scale
        for ax_x, ax_y in pts:
            ax.plot([ax_x, bx], [ax_y, by], color=t["lead"], lw=1.4, zorder=3)
            ax.add_patch(Circle((ax_x, ax_y), 3.2, fc=t["lead"], ec="none", zorder=3))
        ax.add_patch(Circle((bx, by), 17, fc=fc, ec=t["ring"], lw=2.5, zorder=4))
        ax.text(bx, by + 1, num, fontsize=18, fontweight="bold", color=tc, ha="center", va="center", zorder=5)
    kx, ky = 945, 120
    for num, fc, tc, name, body in KEY:
        ax.add_patch(Circle((kx + 17, ky + 13), 17, fc=fc, ec=t["ring"], lw=2, zorder=4))
        ax.text(kx + 17, ky + 14, num, fontsize=18, fontweight="bold", color=tc, ha="center", va="center", zorder=5)
        ax.text(kx + 48, ky, name, fontsize=21.5, fontweight="bold", color=t["title"], va="top")
        ax.text(kx + 48, ky + 31, body, fontsize=18, color=t["body"], va="top", linespacing=1.3)
        ky += 54 + 23.5 * (body.count("\n") + 1)
    ax.text(kx, ky + 6, "4 and 5 form the class <3 sections or split.\nExcluded labels stay in the abundance outputs.",
            fontsize=17, color=t["body"], va="top", linespacing=1.35, style="italic")
    fig.savefig(OUT / f"mitocensus_object_checks_{theme}.png", dpi=144, transparent=True)
    plt.close(fig)
print("ok", crop, round(scale, 3))
