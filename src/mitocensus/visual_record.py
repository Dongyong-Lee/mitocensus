"""Visual record of one MitoCensus analysis.

Every check and measurement is shown on the object, the serial sections and the voxels it came from. The record is
written next to the tables (``visual_record/index.html`` with PNG images) and is built from the same inputs and code as
``objects.csv``: the labels after the minimum-size step, the ROI-normalized image, the section counts, the skeleton
after pruning and the voxel layers returned by the intensity-feature code. For every object page, the features are
recomputed from the displayed layers and compared with the table; the differences are stored in
``visual_record.json``.

Surfaces are marching-cubes renderings of the instance mask (16-nm Gaussian smoothing). They show the segmentation;
they are not membrane reconstructions and carry no new values.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from . import __version__
from .biological_analysis import (
    BiologicalAnalysisConfig,
    _drop_small_pieces,
    _expanded_slice,
    _internal_features,
    _normalize_raw,
    _pruned_skeleton,
)

COLORS = {
    "passing": "#3A5A8C",
    "thin": "#CC79A7",
    "border": "#B8B8B8",
    "disconnected": "#E69F00",
    "split": "#D55E00",
    "ridge": "#009E73",
    "matrix": "#56B4E9",
    "weak_boundary": "#E69F00",
    "rim": "#F0E442",
    "ink": "#222222",
}
CLASSES = ("elongated_fusion_associated", "branched_hyperfused_like", "intermediate", "fragmented_fission_associated")
CLASS_NAMES = {"elongated_fusion_associated": "Elongated", "branched_hyperfused_like": "Branched",
               "intermediate": "Intermediate", "fragmented_fission_associated": "Fragmented"}
CHECKED_FEATURES = ("line_filter_fraction", "matrix_electron_density", "weak_boundary_fraction",
                    "boundary_electron_density")
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
LIGHT = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])


# ----------------------------------------------------------------------------------------------------------------
# data preparation (the same steps as analyze_biological_sample)
# ----------------------------------------------------------------------------------------------------------------

def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else False


def _number(value, default=math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


class _Sample:
    """Dense labels, normalized image and per-object geometry, prepared exactly as the analysis prepares them."""

    def __init__(self, raw: np.ndarray, instances: np.ndarray, spacing: tuple[float, ...], roi: np.ndarray | None,
                 config: BiologicalAnalysisConfig) -> None:
        self.raw = np.asarray(raw)
        self.spacing = tuple(float(v) for v in spacing)
        self.ndim = self.raw.ndim
        self.config = config
        roi_array = np.ones(self.raw.shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
        source_labels = np.where(roi_array, np.asarray(instances), 0)
        source_ids = np.unique(source_labels[source_labels > 0])
        labels = np.zeros(source_labels.shape, dtype=np.uint32)
        foreground = source_labels > 0
        if source_ids.size:
            labels[foreground] = (np.searchsorted(source_ids, source_labels[foreground]) + 1).astype(np.uint32)
        self.source_to_dense = {int(s): d for d, s in enumerate(source_ids.tolist(), start=1)}
        self.labels, _ = _drop_small_pieces(labels, config.min_voxels)
        self.normalized, _ = _normalize_raw(self.raw, roi_array)
        self.slices = ndi.find_objects(self.labels)
        pad = tuple(max(3, int(math.ceil(2 * config.shell_width_nm / v))) for v in self.spacing)
        self.pad = pad

    def item(self, object_id: int):
        dense = self.source_to_dense.get(int(object_id))
        if dense is None or dense > len(self.slices) or self.slices[dense - 1] is None:
            return None, None
        return dense, self.slices[dense - 1]

    def mask(self, object_id: int) -> tuple[np.ndarray | None, tuple | None]:
        dense, item = self.item(object_id)
        if item is None:
            return None, None
        return self.labels[item] == dense, item

    def sections(self, object_id: int) -> list[int]:
        m, item = self.mask(object_id)
        if m is None or self.ndim != 3:
            return []
        zs = np.flatnonzero(m.reshape(m.shape[0], -1).any(axis=1))
        return [int(item[0].start + z) for z in zs]

    def best_section(self, object_id: int) -> int | None:
        m, item = self.mask(object_id)
        if m is None or self.ndim != 3:
            return None
        areas = m.reshape(m.shape[0], -1).sum(axis=1)
        return int(item[0].start + int(np.argmax(areas)))

    def window(self, object_id: int, margin_nm: float = 120.0) -> tuple[slice, slice]:
        """Square in-plane window around the object with a margin."""
        _, item = self.item(object_id)
        ys, xs = (item[-2], item[-1])
        sy, sx = self.spacing[-2], self.spacing[-1]
        cy, cx = (ys.start + ys.stop) / 2, (xs.start + xs.stop) / 2
        half = max((ys.stop - ys.start) * sy, (xs.stop - xs.start) * sx) / 2 + margin_nm
        hy, hx = int(math.ceil(half / sy)), int(math.ceil(half / sx))
        H, W = self.labels.shape[-2:]
        y0, y1 = max(0, int(cy - hy)), min(H, int(cy + hy))
        x0, x1 = max(0, int(cx - hx)), min(W, int(cx + hx))
        return slice(y0, y1), slice(x0, x1)

    def layers(self, object_id: int):
        """Features and voxel layers of one object from the measurement code (same patch as the analysis)."""
        dense, item = self.item(object_id)
        expanded = _expanded_slice(item, self.labels.shape, self.pad)
        object_mask = self.labels[expanded] == dense
        features, layers = _internal_features(self.normalized[expanded], object_mask, self.spacing, self.config,
                                              return_layers=True)
        return features, layers, expanded, object_mask

    def mesh(self, object_id: int, sigma_nm: float = 16.0):
        dense, item = self.item(object_id)
        spacing = np.asarray(self.spacing, dtype=float)
        pad = np.maximum(np.ceil(4 * sigma_nm / spacing).astype(int), 2)
        lo = np.maximum(np.array([s.start for s in item]) - pad, 0)
        hi = np.minimum(np.array([s.stop for s in item]) + pad, self.labels.shape)
        region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        binary = (self.labels[region] == dense).astype(np.float32)
        field = ndi.gaussian_filter(binary, sigma=sigma_nm / spacing, mode="constant", cval=0.0)
        level = min(0.43, 0.5 * float(field.max()))
        if not level > 0:
            return None
        try:
            vertices, faces, _, _ = measure.marching_cubes(field, level=level, spacing=tuple(spacing / 1000.0),
                                                           allow_degenerate=False)
        except (ValueError, RuntimeError):
            return None
        return vertices + lo * spacing / 1000.0, faces


# ----------------------------------------------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3-D projection)

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    return plt


def _shade(triangles: np.ndarray, color) -> np.ndarray:
    from matplotlib.colors import to_rgb

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    intensity = 0.42 + 0.58 * np.clip(np.abs(normals @ LIGHT), 0, 1)
    rgb = np.clip(np.asarray(to_rgb(color))[None, :] * intensity[:, None], 0, 1)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)


def _render(ax, mesh, color, radius_um: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.set_axis_off()
    if mesh is None:
        ax.text2D(0.5, 0.5, "no surface", ha="center", va="center", transform=ax.transAxes, fontsize=7.5)
        return
    vertices, faces = mesh
    xyz = vertices[:, [2, 1, 0]].copy()
    xyz[:, 1] = -xyz[:, 1]
    tri = xyz[faces]
    centre = (tri.reshape(-1, 3).min(axis=0) + tri.reshape(-1, 3).max(axis=0)) / 2
    tri = tri - centre
    collection = Poly3DCollection(tri, facecolors=_shade(tri, color), edgecolors="none", linewidths=0)
    collection.set_rasterized(True)
    ax.add_collection3d(collection)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-radius_um, radius_um)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-58)


def _outline(ax, mask2d: np.ndarray, color: str, lw: float = 0.9) -> None:
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _image(ax, img: np.ndarray) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _scalebar(ax, spacing_xy: float, length_nm: float = 200.0) -> None:
    px = length_nm / spacing_xy
    h, w = ax.images[0].get_array().shape[:2] if ax.images else (1, 1)
    if px < w * 0.6:
        ax.plot([w * 0.06, w * 0.06 + px], [h * 0.92, h * 0.92], color="white", lw=1.6, solid_capstyle="butt")


_DPI = {"value": 150}


def _save(fig, path: Path, dpi: int | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or _DPI["value"], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path.name if path.parent.name != "objects" else f"objects/{path.name}"


# ----------------------------------------------------------------------------------------------------------------
# categories
# ----------------------------------------------------------------------------------------------------------------

def _category(row: dict) -> str:
    reasons = str(row.get("non_evaluable_reason") or "")
    if _truthy(row.get("morphology_evaluable")):
        return "passing"
    if "axial_extent_below_minimum" in reasons or "axial_sections_below_minimum" in reasons:
        return "thin"
    if "border" in reasons:
        return "border"
    if "disconnected" in reasons:
        return "disconnected"
    if "split_label_candidate" in reasons:
        return "split"
    return "border"


def _pick_by_quantile(rows: list[dict], key: str, k: int) -> list[dict]:
    if len(rows) <= k:
        return sorted(rows, key=lambda r: _number(r.get(key), 0.0))
    values = np.log10(np.maximum([_number(r.get(key), 1e-9) for r in rows], 1e-12))
    order = np.argsort(values)
    chosen, used = [], set()
    for q in np.linspace(0.1, 0.9, k):
        target = np.quantile(values, q)
        for i in sorted(order, key=lambda j: abs(values[j] - target)):
            if i not in used:
                used.add(i)
                chosen.append(rows[i])
                break
    return sorted(chosen, key=lambda r: _number(r.get(key), 0.0))


# ----------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------

def _overview(rows: list[dict], sample: _Sample, path: Path) -> str:
    plt = _plt()
    cats = [_category(r) for r in rows]
    flagged = sum(_truthy(r.get("split_label_candidate")) and c == "passing" for r, c in zip(rows, cats))
    names = [("passing", "QC-passing"), ("thin", "Too thin across sections"), ("border", "Touches border / ROI"),
             ("disconnected", "Several pieces"), ("split", "Split label (excluded)")]
    counts = [(label, cats.count(key), COLORS[key]) for key, label in names if cats.count(key) or key == "passing"]
    fig = plt.figure(figsize=(8.0, 2.5))
    ax = fig.add_axes([0.08, 0.18, 0.22, 0.7])
    y = np.arange(len(counts))[::-1]
    ax.barh(y, [c for _, c, _ in counts], color=[col for _, _, col in counts], height=0.6)
    for yy, (label, c, _) in zip(y, counts):
        extra = f" ({flagged} flagged as possible splits)" if label == "QC-passing" and flagged else ""
        ax.text(c, yy, f" {c}{extra}", va="center", fontsize=7.5)
    ax.set_yticks(y, [label for label, _, _ in counts])
    ax.set_xlabel("Labels")
    ax.set_title("Check outcome", loc="left")
    ax2 = fig.add_axes([0.38, 0.18, 0.22, 0.7])
    ax3 = fig.add_axes([0.66, 0.18, 0.18, 0.7])
    if sample.ndim == 3:
        sec = np.array([_number(r.get("axial_sections"), np.nan) for r in rows])
        thin = np.array([c == "thin" for c in cats])
        top = int(np.nanmax(sec)) if np.isfinite(sec).any() else 1
        bins = np.arange(0.5, min(top, 40) + 1.5)
        ax2.hist([np.clip(sec[~thin & np.isfinite(sec)], 0, 40), sec[thin & np.isfinite(sec)]], bins=bins,
                 stacked=True, color=[COLORS["border"], COLORS["thin"]], label=["Other labels", "Too thin"])
        if sample.config.min_axial_extent_nm:
            ax2.axvline(sample.config.min_axial_extent_nm / sample.spacing[0] - 0.5, color=COLORS["ink"], lw=0.7,
                        ls=(0, (2, 1.5)))
        ax2.set_xlabel("Sections spanned")
        ax2.set_ylabel("Labels")
        ax2.legend(frameon=False, loc="upper right")
        ax2.set_title("Extent across sections", loc="left")
        vol = np.array([_number(r.get("physical_size"), np.nan) for r in rows])
        for key in ("border", "disconnected", "split", "thin", "passing"):
            sel = np.array([c == key for c in cats]) & np.isfinite(vol) & np.isfinite(sec)
            if sel.any():
                ax3.scatter(sec[sel], vol[sel], s=5, color=COLORS[key], lw=0, label=dict(names)[key], alpha=0.85)
        ax3.set_xscale("log"), ax3.set_yscale("log")
        ax3.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=1.6, handletextpad=0.2,
                   borderaxespad=0.1)
        ax3.set_xlabel("Sections spanned")
        ax3.set_ylabel("Volume (µm³)")
        ax3.set_title("Size and extent of every label", loc="left")
    else:
        ax2.set_axis_off(), ax3.set_axis_off()
    return _save(fig, path)


def _thin_gallery(rows: list[dict], sample: _Sample, path: Path, k: int) -> tuple[str | None, list[int]]:
    thin = [r for r in rows if _category(r) == "thin"]
    if not thin or sample.ndim != 3:
        return None, []
    thin = sorted(thin, key=lambda r: -_number(r.get("major_axis_um"), 0.0))[:k]
    plt = _plt()
    n = len(thin)
    cols = min(n, 6)
    nrows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(1.3 * cols, 1.5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, r in zip(axes.ravel(), thin):
        oid = int(_number(r["object_id"]))
        z = sample.best_section(oid)
        ys, xs = sample.window(oid)
        dense, _ = sample.item(oid)
        _image(ax, sample.normalized[z][ys, xs])
        _outline(ax, sample.labels[z][ys, xs] == dense, COLORS["thin"])
        _scalebar(ax, sample.spacing[-1])
        ns = int(_number(r.get("axial_sections"), 0))
        ax.set_title(f"ID {oid} · {ns} section{'s' if ns != 1 else ''}\n{_number(r.get('major_axis_um'), 0):.2f} µm long",
                     fontsize=7.5)
    fig.suptitle("Labels excluded as too thin across sections (longest first; outline in the section of largest area; "
                 "bar 200 nm)", fontsize=8.5, x=0.01, ha="left", y=1.0 + 0.32 / nrows)
    return _save(fig, path), [int(_number(r["object_id"])) for r in thin]


def _class_galleries(rows: list[dict], sample: _Sample, path: Path, k: int) -> tuple[str | None, list[int]]:
    """QC-passing objects of each shape class nearest to the 10th-90th percentiles of volume. With at most ``k``
    objects in all, one row with a common scale; otherwise one row (and one scale) per class."""
    passing = [r for r in rows if _category(r) == "passing"]
    groups = [(c, [r for r in passing if r.get("shape_class") == c]) for c in CLASSES]
    groups = [(c, _pick_by_quantile(g, "physical_size", k)) for c, g in groups if g]
    if not groups:
        return None, []
    plt = _plt()
    meshes = {}
    for _, g in groups:
        for r in g:
            oid = int(_number(r["object_id"]))
            meshes[oid] = sample.mesh(oid) if sample.ndim == 3 else None

    def radius_of(objs) -> float:
        ext = [np.ptp(meshes[int(_number(r["object_id"]))][0], axis=0).max() / 2 for r in objs
               if meshes.get(int(_number(r["object_id"]))) is not None]
        return max(ext) * 1.02 if ext else 1.0

    compact = sum(len(g) for _, g in groups) <= k
    lines = [[(c, r) for c, g in groups for r in g]] if compact else [[(c, r) for r in g] for c, g in groups]
    ncols = max(len(line) for line in lines)
    pad = 0.15 if compact else 1.05
    head = 0.3 if compact else 0.0
    row_h = 1.4
    W, H = 1.25 * ncols + pad, head + row_h * len(lines)

    def axes(x, y_top, w, h, **kw):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H], **kw)

    fig = plt.figure(figsize=(W, H))
    shown = []
    for gi, line in enumerate(lines):
        radius = radius_of([r for _, r in line])
        top = head + gi * row_h
        if not compact:
            fig.text(0.05 / W, 1 - (top + 0.55) / H, f"{CLASS_NAMES.get(line[0][0], line[0][0])}\n"
                     f"(half-width {radius:.2f} µm)", fontsize=8, va="center", ha="left")
        previous = None
        for j, (c, r) in enumerate(line):
            oid = int(_number(r["object_id"]))
            shown.append(oid)
            x = pad + 1.25 * j
            if compact and c != previous:
                fig.text((x + 0.05) / W, 1 - (top - 0.05) / H, CLASS_NAMES.get(c, c), fontsize=8, va="bottom",
                         ha="left", weight="bold")
            previous = c
            color = COLORS["split"] if _truthy(r.get("split_label_candidate")) else COLORS["passing"]
            if sample.ndim == 3:
                ax = axes(x, top, 1.18, 0.95, projection="3d")
                ax.computed_zorder = False
                _render(ax, meshes[oid], color, radius)
            else:
                ax = axes(x, top, 1.18, 0.95)
                ys, xs = sample.window(oid)
                dense, _ = sample.item(oid)
                _image(ax, sample.normalized[ys, xs])
                _outline(ax, sample.labels[ys, xs] == dense, color)
            ns = int(_number(r.get("axial_sections"), 0))
            text = f"ID {oid}\n{_number(r.get('physical_size'), 0):.2g} µm³" + (f" · {ns} sections" if ns else "")
            if _truthy(r.get("split_label_candidate")):
                text += "\nflag: possible split"
            fig.text((x + 0.59) / W, 1 - (top + 0.97) / H, text, ha="center", va="top", fontsize=7.5)
    scale = (f"same scale, half-width {radius_of([r for line in lines for _, r in line]):.2f} µm" if compact
             else "one scale per row")
    fig.suptitle(f"QC-passing objects by shape class, nearest to the 10th–90th percentiles of volume ({scale})",
                 fontsize=8.5, x=0.01, ha="left", y=1.0 + 0.32 / H)
    return _save(fig, path), shown


def _split_figure(rows: list[dict], summary: dict, sample: _Sample, path: Path, k: int) -> tuple[str | None, list[int]]:
    groups = (summary.get("quality_control_v2") or {}).get("split_label_groups") or []
    if not groups or sample.ndim != 3:
        return None, []
    groups = groups[:k]
    plt = _plt()
    fig, axes = plt.subplots(len(groups), 2, figsize=(3.2, 1.65 * len(groups)), squeeze=False)
    shown = []
    for gi, group in enumerate(groups):
        a, b = int(group[0]), int(group[1])
        shown += [a, b]
        sa, sb = sample.sections(a), sample.sections(b)
        if not sa or not sb:
            continue
        za, zb = (max(sa), min(sb)) if max(sa) < min(sb) else (min(sa), max(sb))
        da, _ = sample.item(a)
        db, _ = sample.item(b)
        ys, xs = sample.window(a, margin_nm=200.0)
        for ax, z, title in ((axes[gi, 0], za, f"section {za}"), (axes[gi, 1], zb, f"section {zb}")):
            _image(ax, sample.normalized[z][ys, xs])
            _outline(ax, sample.labels[z][ys, xs] == da, COLORS["passing"])
            _outline(ax, sample.labels[z][ys, xs] == db, COLORS["split"])
            ax.set_title(title, fontsize=7.5)
        axes[gi, 0].set_ylabel(f"IDs {a} → {b}", fontsize=7.5)
    fig.suptitle("Possible splits: one label ends (blue) and another continues (vermillion) within the gap allowed",
                 fontsize=8.5, x=0.01, ha="left")
    return _save(fig, path), shown


REVIEW_FLAGS = (("megamitochondrion_candidate", "Megamitochondrion"), ("ring_or_fenestrated_candidate", "Ring or fenestrated"),
                ("spheroid_like_candidate", "Spheroid-like"), ("moas_like_candidate", "MOAS-like"),
                ("nanotunnel_like_candidate", "Nanotunnel-like"), ("swelling_like_candidate", "Swelling-like"),
                ("constriction_site_candidate", "Constriction"), ("composite_review_candidate", "Composite review"),
                ("split_label_candidate", "Possible split"), ("manual_review_recommended", "Any review flag"))


def _summary_figure(rows: list[dict], summary: dict, sample: _Sample, path: Path) -> str:
    """Amount per section, size of QC-passing objects, spacing against random placement and review flags."""
    plt = _plt()
    fn = summary.get("functions") or {}
    passing = [r for r in rows if _category(r) == "passing"]
    fig = plt.figure(figsize=(8.0, 2.4))
    ax1 = fig.add_axes([0.05, 0.2, 0.19, 0.66])
    if sample.ndim == 3:
        frac = (sample.labels > 0).reshape(sample.labels.shape[0], -1).mean(axis=1)
        ax1.plot(np.arange(len(frac)), frac * 100, color=COLORS["passing"], lw=0.9)
        total = _number((fn.get("mitochondrial_amount") or {}).get("mitochondrial_fraction"))
        if math.isfinite(total):
            ax1.axhline(total * 100, color=COLORS["ink"], lw=0.6, ls=(0, (2, 1.5)))
        ax1.set_xlabel("Section")
        ax1.set_ylabel("Labelled area (%)")
        ax1.set_title("Amount per section", loc="left")
    else:
        ax1.set_axis_off()
    ax2 = fig.add_axes([0.31, 0.2, 0.19, 0.66])
    sizes = np.array([_number(r.get("physical_size")) for r in passing], dtype=float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if sizes.size:
        bins = np.logspace(np.log10(sizes.min()) - 0.05, np.log10(sizes.max()) + 0.05, 14)
        ax2.hist(sizes, bins=bins, color=COLORS["passing"])
        ax2.set_xscale("log")
        from matplotlib.ticker import LogLocator, NullFormatter
        ax2.xaxis.set_major_locator(LogLocator(base=10))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.axvline(np.median(sizes), color=COLORS["ink"], lw=0.7)
        edge = summary.get("edge_correction") or {}
        if edge.get("enabled") and math.isfinite(_number(edge.get("weighted_mean_physical_size"))):
            ax2.axvline(_number(edge.get("weighted_mean_physical_size")), color=COLORS["split"], lw=0.7,
                        ls=(0, (2, 1.5)))
    unit = "µm³" if sample.ndim == 3 else "µm²"
    ax2.set_xlabel(f"Size of QC-passing objects ({unit})")
    ax2.set_ylabel("Objects")
    ax2.set_title(f"Size (n = {sizes.size})", loc="left")
    ax3 = fig.add_axes([0.57, 0.2, 0.17, 0.66])
    nn = np.array([_number(r.get("nearest_neighbor_um")) for r in passing], dtype=float)
    nn = nn[np.isfinite(nn)]
    if nn.size:
        ax3.hist(nn, bins=12, color=COLORS["passing"])
        csr = _number((fn.get("spatial_organization") or {}).get("csr_expected_nearest_neighbor_um"))
        if math.isfinite(csr):
            ax3.axvline(csr, color=COLORS["ink"], lw=0.7, ls=(0, (2, 1.5)))
    ax3.set_xlabel("Nearest-neighbour distance (µm)")
    ax3.set_ylabel("Objects")
    ax3.set_title("Spacing", loc="left")
    ax4 = fig.add_axes([0.85, 0.2, 0.13, 0.66])
    counts = [(label, sum(_truthy(r.get(key)) for r in passing)) for key, label in REVIEW_FLAGS]
    y = np.arange(len(counts))[::-1]
    ax4.barh(y, [c for _, c in counts], color=[COLORS["split"] if label == "Possible split" else COLORS["border"]
                                               for label, _ in counts], height=0.6)
    for yy, (_, c) in zip(y, counts):
        ax4.text(c, yy, f" {c}", va="center", fontsize=7)
    ax4.set_yticks(y, [label for label, _ in counts], fontsize=7)
    ax4.set_xlabel("QC-passing objects")
    ax4.set_title("Review flags", loc="left")
    return _save(fig, path)


def _object_page(row: dict, sample: _Sample, path: Path) -> dict:
    plt = _plt()
    oid = int(_number(row["object_id"]))
    dense, item = sample.item(oid)
    features, layers, expanded, object_mask = sample.layers(oid)
    check = {f: (abs(_number(features.get(f)) - _number(row.get(f))) if math.isfinite(_number(features.get(f)))
                 and math.isfinite(_number(row.get(f))) else (0.0 if not math.isfinite(_number(features.get(f)))
                 and not math.isfinite(_number(row.get(f))) else math.inf)) for f in CHECKED_FEATURES}
    isotropic, step, skeleton, _, _ = _pruned_skeleton(object_mask, sample.spacing, sample.config)
    skeleton_match = int(skeleton.sum()) == int(_number(row.get("skeleton_voxels"), -1))
    category = _category(row)
    color = COLORS["split"] if _truthy(row.get("split_label_candidate")) else COLORS.get(category, COLORS["passing"])
    fig = plt.figure(figsize=(7.8, 3.6))
    ys, xs = sample.window(oid)
    if sample.ndim == 3:
        secs = sample.sections(oid)
        pick = secs if len(secs) <= 8 else [secs[int(round(i))] for i in np.linspace(0, len(secs) - 1, 8)]
        for j, z in enumerate(pick):
            ax = fig.add_axes([0.01 + j * 0.105, 0.56, 0.098, 0.38])
            _image(ax, sample.normalized[z][ys, xs])
            _outline(ax, sample.labels[z][ys, xs] == dense, color)
            ax.set_title(f"section {z}", fontsize=7.5)
        ax = fig.add_axes([0.0, 0.03, 0.2, 0.46], projection="3d")
        ax.computed_zorder = False
        m = sample.mesh(oid)
        _render(ax, m, color, (np.ptp(m[0], axis=0).max() / 2 * 1.05) if m is not None else 1.0)
        ax2 = fig.add_axes([0.22, 0.05, 0.17, 0.42])
        proj = isotropic.max(axis=0)
        ax2.imshow(proj, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
        sk = np.argwhere(skeleton)
        if len(sk):
            ax2.scatter(sk[:, 2], sk[:, 1], s=1.2, color=COLORS["ink"], lw=0)
        ax2.set_axis_off()
        ax2.set_title("Skeleton (projected across sections)", fontsize=7.5)
        zbest = sample.best_section(oid)
        zloc = zbest - expanded[0].start
        layer2d = {k: v[zloc] for k, v in layers.items()}
        base = sample.normalized[expanded][zloc]
    else:
        ax = fig.add_axes([0.01, 0.1, 0.3, 0.8])
        _image(ax, sample.normalized[ys, xs])
        _outline(ax, sample.labels[ys, xs] == dense, color)
        layer2d = layers
        base = sample.normalized[expanded]
        zbest = None
    for j, (key, title) in enumerate((("line_filter_ridge", "Line-filter voxels"), ("matrix", "Matrix voxels"),
                                      ("weak_boundary", "Weak-boundary voxels"))):
        ax = fig.add_axes([0.41 + j * 0.15, 0.05, 0.14, 0.42])
        _image(ax, base)
        overlay = np.zeros(base.shape + (4,))
        from matplotlib.colors import to_rgb

        rgb = to_rgb(COLORS["ridge" if key == "line_filter_ridge" else key])
        overlay[layer2d[key]] = (*rgb, 0.85)
        ax.imshow(overlay, interpolation="nearest")
        _outline(ax, object_mask[zloc] if sample.ndim == 3 else object_mask, COLORS["rim"], lw=0.5)
        ax.set_title(title, fontsize=7.5)
    values = [("Volume", f"{_number(row.get('physical_size')):.3g} µm³" if sample.ndim == 3
               else f"{_number(row.get('physical_size')):.3g} µm²"),
              ("Sections", f"{int(_number(row.get('axial_sections'), 0))}" if sample.ndim == 3 else "—"),
              ("Major / minor axis", f"{_number(row.get('major_axis_um')):.2f} / {_number(row.get('minor_axis_um')):.2f} µm"),
              ("Elongation", f"{_number(row.get('elongation')):.2f}"),
              ("Shape class", str(row.get("shape_class"))),
              ("Line-filter fraction", f"{_number(row.get('line_filter_fraction')):.3f}"),
              ("Matrix density", f"{_number(row.get('matrix_electron_density')):.3f}"),
              ("Weak boundary", f"{_number(row.get('weak_boundary_fraction')):.3f}"),
              ("Excluded because", str(row.get("non_evaluable_reason") or "—")),
              ("Split flag", "yes" if _truthy(row.get("split_label_candidate")) else "no")]
    tx = fig.add_axes([0.87, 0.05, 0.13, 0.9])
    tx.set_axis_off()
    tx.text(0, 1, f"ID {oid}", fontsize=9.5, weight="bold", va="top")
    for i, (k, v) in enumerate(values):
        tx.text(0, 0.9 - i * 0.085, f"{k}\n{v}", fontsize=7.5, va="top")
    fig.suptitle(f"Object {oid}: sections (outline), surface, skeleton and the voxels behind the intensity features"
                 + (f" (section {zbest})" if zbest is not None else ""), fontsize=8.5, x=0.01, ha="left", y=1.0)
    name = _save(fig, path)
    return {"object_id": oid, "image": name, "feature_abs_difference": check, "skeleton_voxels_match": skeleton_match}


# ----------------------------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------------------------

CSS = """body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;max-width:1180px;color:#222;
background:#fff;line-height:1.45}h1{font-size:20px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd}
img{max-width:100%;border:1px solid #eee}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;
padding:3px 8px;text-align:left}.note{color:#555;font-size:13px}.grid a{display:inline-block;margin:2px 6px 2px 0}"""


def _html(title: str, blocks: list[str]) -> str:
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1\"><title>" + html.escape(title) + "</title><style>" + CSS
            + "</style></head><body>" + "\n".join(blocks) + "</body></html>\n")


# ----------------------------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------------------------

def write_visual_record(raw: np.ndarray, instances: np.ndarray, spacing_nm: Sequence[float], rows: list[dict],
                        summary: dict, output: Path, *, sample_id: str, config: BiologicalAnalysisConfig | None = None,
                        roi: np.ndarray | None = None, gallery_size: int = 5, max_thin: int = 12,
                        max_split_groups: int = 6, max_object_pages: int = 24, dpi: int = 150) -> dict:
    """Write ``output`` (index.html, PNG images, visual_record.json) for one analysed sample."""

    _DPI["value"] = int(dpi)
    config = config or BiologicalAnalysisConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample = _Sample(raw, instances, tuple(spacing_nm), roi, config)
    rows = [r for r in rows if sample.item(int(_number(r["object_id"])))[1] is not None]
    files = {"overview": _overview(rows, sample, output / "overview.png"),
             "summary": _summary_figure(rows, summary, sample, output / "sample_summary.png")}
    thin_png, thin_ids = _thin_gallery(rows, sample, output / "thin_labels.png", max_thin)
    class_png, class_ids = _class_galleries(rows, sample, output / "shape_classes.png", gallery_size)
    split_png, split_ids = _split_figure(rows, summary, sample, output / "split_flags.png", max_split_groups)
    by_id = {int(_number(r["object_id"])): r for r in rows}
    page_ids = list(dict.fromkeys(class_ids + split_ids + thin_ids[:3]))[:max_object_pages]
    pages = [_object_page(by_id[i], sample, output / "objects" / f"object_{i}.png") for i in page_ids if i in by_id]
    worst = max((max(p["feature_abs_difference"].values()) for p in pages), default=0.0)
    consistency = {"objects_checked": len(pages), "max_abs_feature_difference": worst,
                   "features": list(CHECKED_FEATURES),
                   "all_skeleton_voxel_counts_match": all(p["skeleton_voxels_match"] for p in pages)}
    cats = [_category(r) for r in rows]
    qc = summary.get("quality_control_v2") or {}
    counts = [("Labels measured", len(rows)), ("QC-passing", cats.count("passing")),
              ("… flagged as possible splits", sum(_truthy(r.get("split_label_candidate")) and c == "passing"
                                                     for r, c in zip(rows, cats))),
              ("Too thin across sections", cats.count("thin")), ("Touch the border or ROI", cats.count("border")),
              ("Several pieces", cats.count("disconnected")), ("Split labels excluded", cats.count("split"))]
    settings = [("Spacing (nm, " + ("z, y, x" if sample.ndim == 3 else "y, x") + ")",
                 ", ".join(f"{v:g}" for v in sample.spacing)),
                ("Minimum extent across sections", f"{config.min_axial_extent_nm:g} nm" if config.min_axial_extent_nm
                 else "off"),
                ("Minimum sections", str(config.min_axial_sections) if config.min_axial_sections else "off"),
                ("Split screen", f"IoU ≥ {config.split_screen_min_iou:g}, gap ≤ {config.split_screen_max_gap_sections}"
                 f" sections, action: {config.split_screen_action}" if config.split_screen_min_iou else "off"),
                ("Minimum label size", f"{config.min_voxels} voxels")]
    blocks = [f"<h1>MitoCensus visual record — {html.escape(sample_id)}</h1>",
              f"<p class=note>MitoCensus {__version__}. Every image is drawn from the labels and image the analysis "
              "measured; numbers are those of <code>objects.csv</code>. Surfaces are marching-cubes renderings of the "
              "mask (16-nm smoothing) and carry no new values.</p>",
              "<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in counts) + "</table>",
              "<h2>Settings</h2><table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                                                    for k, v in settings) + "</table>",
              "<h2>1. Check outcome</h2>", f"<img src=\"{files['overview']}\" alt=\"check overview\">",
              "<p class=note>Labels that span too few sections cannot support whole-object shape measures: their "
              "extent across sections reflects the sampling, not the object. They stay in abundance summaries.</p>"]
    blocks += ["<h2>Amount, size, spacing and review flags</h2>", f"<img src=\"{files['summary']}\" alt=\"sample summary\">",
               "<p class=note>Dashed lines: the whole region (amount) and the distance expected for randomly placed "
               "objects (spacing); solid line: median size. Size, spacing and flags are computed from QC-passing "
               "objects only. A single small region holds few whole objects; pool regions before interpreting "
               "sizes.</p>"]
    if thin_png:
        files["thin_labels"] = thin_png
        blocks += ["<h2>2. Labels excluded as too thin</h2>", f"<img src=\"{thin_png}\" alt=\"thin labels\">",
                   "<p class=note>Check that these are fragments or caps rather than whole mitochondria.</p>"]
    if class_png:
        files["shape_classes"] = class_png
        blocks += ["<h2>3. QC-passing objects by shape class</h2>", f"<img src=\"{class_png}\" alt=\"shape classes\">",
                   "<p class=note>Classes are relative to the reference profile "
                   f"({html.escape(str((summary.get('calibration') or {}).get('source', 'sample')))}) and describe "
                   "static geometry.</p>"]
    if split_png:
        files["split_flags"] = split_png
        blocks += ["<h2>4. Possible splits flagged for review</h2>", f"<img src=\"{split_png}\" alt=\"split flags\">",
                   f"<p class=note>{qc.get('split_label_flagged_objects', 0)} labels in "
                   f"{len(qc.get('split_label_groups') or [])} groups. Labels are never merged or re-measured.</p>"]
    if pages:
        blocks += ["<h2>5. Object pages</h2>",
                   "<p class=note>Each page shows the sections an object spans with its outline, its surface, its "
                   "skeleton after pruning and, in the section of largest area, the voxels counted by the line-filter, "
                   "matrix and weak-boundary measures. The features recomputed from these voxels differ from the table "
                   f"by at most {worst:.2g}.</p>"]
        blocks += [f"<h3 id=o{p['object_id']}>Object {p['object_id']}</h3><img src=\"{p['image']}\" "
                   f"alt=\"object {p['object_id']}\">" for p in pages]
    (output / "index.html").write_text(_html(f"MitoCensus visual record — {sample_id}", blocks), encoding="utf-8")
    record = {"schema_version": "mitocensus.visual-record.v1", "mitocensus": __version__, "sample_id": sample_id,
              "files": dict(files, index="index.html", objects=[p["image"] for p in pages]),
              "objects_shown": {"thin_labels": thin_ids, "shape_classes": class_ids, "split_flags": split_ids,
                                "object_pages": [p["object_id"] for p in pages]},
              "consistency": consistency, "pages": pages}
    (output / "visual_record.json").write_text(json.dumps(record, indent=1, default=float) + "\n", encoding="utf-8")
    return record
