"""MitoCensus workflow as a numbered timeline of cards (light and dark variants, PNG + SVG).

Content follows README.md, docs/QUICKSTART.md, docs/BIOLOGICAL_ANALYSIS.md and the manuscript (four interpretation
classes). Units: 1 pt = 1 px (dpi 72)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none"})
MONO = "Menlo"
SEM = {"qc": "#1F5F99", "border": "#C3C9D0", "disc": "#7A838C", "thin": "#CC79A7"}
THEMES = {
    "light": dict(card="#FFFFFF", edge="#D0D7DE", shadow="#1F2328", title="#1F2328", body="#57606A", line="#D0D7DE",
                  pill="#F6F8FA", chipedge="#1F2328",
                  acc=["#64748B", "#1F5F99", "#0F766E", "#6D28D9", "#B45309"]),
    "dark": dict(card="#161B22", edge="#30363D", shadow="#000000", title="#F0F6FC", body="#9198A1", line="#30363D",
                 pill="#0D1117", chipedge="#F0F6FC",
                 acc=["#94A3B8", "#4C9BE8", "#2DD4BF", "#A78BFA", "#F5A524"]),
}
W = 1200
X0, X1 = 100, 1180          # card extent
CONTENT = 540               # content column
COL2 = 310                  # offset of the second content column
GAP = 24


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=14, ls="-", z=1, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z, alpha=alpha))


def txt(ax, x, y, s, color, size=17, weight="normal", family=None, va="top", ha="left", ls=1.32):
    kw = dict(fontsize=size, color=color, fontweight=weight, va=va, ha=ha, linespacing=ls, zorder=5)
    if family:
        kw["family"] = family
    ax.text(x, y, s, **kw)


# ------------------------------------------------------------------ icons (64 x 64 box at x, y)
def icon_stack(ax, x, y, c, t):
    for k in range(3):
        ax.add_patch(Polygon([(x + 8 + 6 * k, y + 40 - 10 * k), (x + 40 + 6 * k, y + 40 - 10 * k),
                              (x + 52 + 6 * k, y + 26 - 10 * k), (x + 20 + 6 * k, y + 26 - 10 * k)], closed=True,
                             fc=t["card"], ec=c, lw=2.2, zorder=6))
    ax.add_patch(Circle((x + 38, y + 13), 5.5, fc=c, ec="none", zorder=7))


def icon_check(ax, x, y, c, t):
    ax.add_patch(Circle((x + 32, y + 32), 25, fc="none", ec=c, lw=2.6, zorder=6))
    ax.plot([x + 20, x + 29, x + 45], [y + 33, y + 42, y + 22], color=c, lw=3.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=7)


def icon_bars(ax, x, y, c, t):
    for k, h in enumerate((20, 34, 26, 44)):
        ax.add_patch(Rectangle((x + 8 + 12.5 * k, y + 56 - h), 9, h, fc=c, ec="none", zorder=6, alpha=0.35 + 0.15 * k))
    ax.plot([x + 4, x + 60], [y + 57, y + 57], color=c, lw=2.2, zorder=7)


def icon_table(ax, x, y, c, t):
    rbox(ax, x + 6, y + 10, 52, 44, "none", c, lw=2.4, r=5, z=6)
    ax.add_patch(Rectangle((x + 6, y + 10), 52, 11, fc=c, ec="none", zorder=6, alpha=0.5))
    for k in (1, 2):
        ax.plot([x + 6 + 17.3 * k] * 2, [y + 10, y + 54], color=c, lw=1.6, zorder=7)
    for k in (1, 2):
        ax.plot([x + 6, x + 58], [y + 21 + 11 * k] * 2, color=c, lw=1.6, zorder=7)


def icon_groups(ax, x, y, c, t):
    import random
    rnd = random.Random(3)
    for gx, mu in ((x + 18, y + 36), (x + 46, y + 24)):
        for _ in range(7):
            ax.add_patch(Circle((gx + rnd.uniform(-7, 7), mu + rnd.uniform(-11, 11)), 2.6, fc=c, ec="none", zorder=6,
                                alpha=0.75))
        ax.plot([gx - 10, gx + 10], [mu, mu], color=c, lw=3, zorder=7)
    ax.plot([x + 8, x + 8, x + 56, x + 56], [y + 58, y + 61, y + 61, y + 58], color=c, lw=1.8, zorder=7)


ICONS = [icon_stack, icon_check, icon_bars, icon_table, icon_groups]


def chip(ax, x, y, color, label, t):
    ax.add_patch(Rectangle((x, y - 8), 17, 17, fc=color, ec=t["chipedge"], lw=0.7, zorder=6))
    txt(ax, x + 27, y + 1, label, t["title"], size=17.5, va="center")


def text_w(s, size, bold=False):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial", weight="bold" if bold else "normal")).get_extents().width


def optional(ax, y, label, body, c, t):
    x0, x1 = CONTENT - 8, X1 - 20
    rbox(ax, x0, y, x1 - x0, 40, t["pill"], c, lw=1.4, r=20, ls=(0, (5, 3)), z=4)
    txt(ax, x0 + 20, y + 21, label, c, size=16.5, weight="bold", va="center")
    bx = x0 + 20 + text_w(label, 16.5, True) + 14
    assert bx + text_w(body, 16) < x1 - 16, (label, body)
    txt(ax, bx, y + 21, body, t["body"], size=16, va="center")


def draw(theme):
    t = THEMES[theme]
    heights = [196, 176, 262, 128, 118]
    H = 24 + sum(heights) + GAP * (len(heights) - 1) + 24
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    tops, y = [], 24
    for h in heights:
        tops.append(y); y += h + GAP
    # timeline
    ax.plot([48, 48], [tops[0] + 40, tops[-1] + 40], color=t["line"], lw=3, zorder=0, solid_capstyle="round")
    titles = ["Input", "Object checks", "13 measurement functions", "Object record", "Cohort comparison"]
    subs = ["EM volume and instance mask", "mitocensus analyze-biological", "four interpretation classes",
            "one record per sample", "mitocensus compare-biological"]
    mono = [False, True, False, False, True]
    for i, (top, h) in enumerate(zip(tops, heights)):
        c = t["acc"][i]
        rbox(ax, X0 + 3, top + 4, X1 - X0, h, t["shadow"], "none", lw=0, z=0, alpha=0.06)
        rbox(ax, X0, top, X1 - X0, h, t["card"], t["edge"], z=1)
        ax.add_patch(Rectangle((X0 + 1, top + 14), 5, h - 28, fc=c, ec="none", zorder=2))
        ax.add_patch(Circle((48, top + 40), 21, fc=c, ec=t["card"], lw=3, zorder=3))
        txt(ax, 48, top + 41, str(i + 1), "#FFFFFF" if theme == "light" else "#0D1117", size=20, weight="bold",
            va="center", ha="center")
        ICONS[i](ax, X0 + 24, top + 20, c, t)
        txt(ax, X0 + 104, top + 22, titles[i], t["title"], size=24, weight="bold")
        txt(ax, X0 + 104, top + 56, subs[i], c, size=16 if mono[i] else 17, family=MONO if mono[i] else None)

    # 1 input
    top = tops[0]
    txt(ax, CONTENT, top + 22, "EM volume", t["title"], size=18.5, weight="bold")
    txt(ax, CONTENT, top + 50, "serial sections or a\n2-D TEM field", t["body"], size=16.5)
    txt(ax, CONTENT + COL2, top + 22, "Instance mask", t["title"], size=18.5, weight="bold")
    txt(ax, CONTENT + COL2, top + 50, "aligned labels from\nany segmenter", t["body"], size=16.5)
    optional(ax, top + 136, "Optional refinement", "MitoNet + OmniEM priors into one mask", t["acc"][0], t)

    # 2 checks
    top = tops[1]
    for (key, lab), (dx, dy) in zip((("qc", "QC-passing"), ("border", "Border-truncated"), ("disc", "Disconnected"),
                                     ("thin", "<3 sections or split")), ((0, 32), (COL2, 32), (0, 70), (COL2, 70))):
        chip(ax, CONTENT + dx, top + dy, SEM[key], lab, t)
    txt(ax, CONTENT, top + 104, "Section support and the split screen are opt-in.\nExcluded labels stay in the abundance outputs.",
        t["body"], size=15.5)

    # 3 functions
    top = tops[2]
    blocks = [("Direct", "amount, size, shape,\nheterogeneity, spatial"), ("Intensity-based", "line filter, matrix,\nboundary"),
              ("Review candidates", "rare shapes, strings, close\npairs, stress-associated"), ("Cohort framework", "disease or treatment\nresponse")]
    for k, (head, body) in enumerate(blocks):
        x = CONTENT + (k % 2) * COL2
        yy = top + 22 + (k // 2) * 92
        txt(ax, x, yy, head, t["title"], size=18.5, weight="bold")
        txt(ax, x, yy + 28, body, t["body"], size=16.5)
    optional(ax, top + 204, "Optional edge weights", "Miles–Lantuéjoul, pooled over regions", t["acc"][2], t)

    # 4 record
    top = tops[3]
    txt(ax, CONTENT, top + 22, "objects.csv", t["title"], size=18.5, weight="bold", family=MONO)
    txt(ax, CONTENT, top + 50, "one row per label: checks,\nmeasurements, sections, flags", t["body"], size=16.5)
    txt(ax, CONTENT + COL2, top + 22, "sample_summary.json", t["title"], size=17.5, weight="bold", family=MONO)
    txt(ax, CONTENT + COL2, top + 50, "13 functions with their\ndenominators; report", t["body"], size=16.5)

    # 5 cohort
    top = tops[4]
    txt(ax, CONTENT, top + 24, "Biological-unit endpoints", t["title"], size=18.5, weight="bold")
    txt(ax, CONTENT, top + 54, "control-referenced effect sizes, bootstrap intervals", t["body"], size=16.5)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"mitocensus_workflow_{theme}.{ext}", dpi=144 if ext == "png" else 72, transparent=True)
    plt.close(fig)


for th in THEMES:
    draw(th)
print("ok")
