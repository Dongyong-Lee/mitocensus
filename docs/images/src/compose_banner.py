"""Title and tagline over the Blender banner render (one variant; the banner carries its own dark background)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
img = np.asarray(Image.open(SRC / "banner_render.png").convert("RGB")).astype(float) / 255
H, W = img.shape[:2]
# darken the left side slightly so the title always reads
ramp = np.clip((np.arange(W) - 120) / 600, 0, 1) ** 1.4
img = img * (0.55 + 0.45 * ramp)[None, :, None]

fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
ax.imshow(img, extent=(0, W, H, 0), interpolation="lanczos")
ax.text(72, 150, "MitoCensus", fontsize=88, fontweight="bold", color="#FFFFFF", va="baseline")
ax.text(76, 206, "Object-level quality control for mitochondrial\nmorphometry in volume electron microscopy",
        fontsize=26, color="#B9CBE2", va="top", linespacing=1.4)


from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402


def pill(x, y, label):
    w = TextPath((0, 0), label, size=21, prop=FontProperties(family="Arial")).get_extents().width + 40
    ax.add_patch(FancyBboxPatch((x, y), w, 46, boxstyle="round,pad=0,rounding_size=23", fc="#13213A", ec="#36598C",
                                lw=1.6, zorder=3))
    ax.text(x + w / 2, y + 24, label, fontsize=21, color="#DCE8F7", ha="center", va="center", zorder=4)
    return x + w + 14


x = pill(76, 318, "Object checks")
pill(x, 318, "13 measurement functions")
pill(76, 378, "Biological-unit statistics")
fig.savefig(OUT / "mitocensus_banner.png", dpi=144)
plt.close(fig)
# the README copy is 1.5 times the render size (2400 x 840), downsampled to keep the file small
Image.open(OUT / "mitocensus_banner.png").convert("RGB").resize((W * 3 // 2, H * 3 // 2), Image.LANCZOS).save(OUT / "mitocensus_banner.png")
print("written", OUT / "mitocensus_banner.png")
