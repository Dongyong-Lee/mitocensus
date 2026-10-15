"""Object checks on synthetic shapes in a crop of serial sections (README figure, banner style).

Each object is drawn with glowing cross-section rings where it meets a section, so the number of sections it spans is
visible. Run: Blender -b --factory-startup -P blender_checks_v2.py -- OUTDIR
Writes OUTDIR/checks_render.png (transparent) and OUTDIR/checks_anchors.json (badge positions in pixels)."""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

OUT = Path(sys.argv[sys.argv.index("--") + 1]); OUT.mkdir(parents=True, exist_ok=True)
RES = (1600, 1100)
HX, HY, HZ = 2.0, 2.0, 1.2
SECTIONS = [-HZ + (k + 0.5) * (2 * HZ / 8) for k in range(8)]     # eight serial sections


def lin(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c) + (1.0,)


def mat(name, hexcol, rough=0.38, alpha=1.0, emit=0.0, coat=0.0, subsurf=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin(hexcol); b.inputs["Roughness"].default_value = rough
    b.inputs["Alpha"].default_value = alpha
    if emit:
        b.inputs["Emission Color"].default_value = lin(hexcol); b.inputs["Emission Strength"].default_value = emit
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if subsurf:
        b.inputs["Subsurface Weight"].default_value = subsurf; b.inputs["Subsurface Radius"].default_value = (0.25, 0.25, 0.25)
    return m


def metaball(name, elements, res=0.045):
    mb = bpy.data.metaballs.new(name); mb.resolution, mb.render_resolution = res, res / 2
    o = bpy.data.objects.new(name, mb); bpy.context.collection.objects.link(o)
    for typ, co, r, sx, sy, sz in elements:
        e = mb.elements.new(type=typ); e.co = co; e.radius = r
        if typ == "ELLIPSOID":
            e.size_x, e.size_y, e.size_z = sx, sy, sz
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.convert(target="MESH"); m = bpy.context.view_layer.objects.active
    bpy.ops.object.shade_smooth()
    return m


def tube(name, pts, r, n=22):
    els = []
    for i in range(len(pts) - 1):
        a, b = Vector(pts[i]), Vector(pts[i + 1])
        for k in range(n):
            els.append(("BALL", a.lerp(b, k / n), r * (0.94 + 0.12 * math.sin(2.7 * (i + k / n))), 0, 0, 0))
    els.append(("BALL", Vector(pts[-1]), r, 0, 0, 0))
    return metaball(name, els)


def box_obj(name, scale, loc):
    me = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(o)
    o.scale = scale; o.location = loc
    return o


def bake(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    return me


def rings(src, ring_mat):
    for z in SECTIONS:
        cut = box_obj("slab", (12, 12, 0.026), (0, 0, z))
        tmp = src.copy(); tmp.data = src.data.copy(); bpy.context.collection.objects.link(tmp)
        for md in list(tmp.modifiers):
            pass
        m = tmp.modifiers.new("x", "BOOLEAN"); m.operation = "INTERSECT"; m.object = cut; m.solver = "EXACT"
        me = bake(tmp)
        bpy.data.objects.remove(tmp); bpy.data.objects.remove(cut)
        if len(me.vertices):
            s = bpy.data.objects.new("ring", me); bpy.context.collection.objects.link(s)
            s.data.materials.clear(); s.data.materials.append(ring_mat); s.scale = (1.015, 1.015, 1.0)


def polyline(pts, m, bevel):
    cu = bpy.data.curves.new("l", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = bevel; cu.bevel_resolution = 2
    sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q[0], q[1], q[2], 1.0)
    o = bpy.data.objects.new("l", cu); bpy.context.collection.objects.link(o); cu.materials.append(m)


# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 160; sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = RES
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

# sections as faint sheets with a thin rim, and the crop box
m_sheet = mat("sheet", "#8FA9C8", rough=0.3, alpha=0.018)
m_rim = mat("rim", "#9AA9BC", emit=0.5)
m_box = mat("box", "#8C959F", emit=0.6)
for z in SECTIONS:
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, z)); p = bpy.context.active_object
    p.scale = (2 * HX, 2 * HY, 1); p.data.materials.append(m_sheet)
    polyline([(-HX, -HY, z), (HX, -HY, z), (HX, HY, z), (-HX, HY, z), (-HX, -HY, z)], m_rim, 0.0045)
cs = [Vector((sx * HX, sy * HY, sz * HZ)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
for i, a in enumerate(cs):
    for b in cs[i + 1:]:
        if sum(1 for u, v in zip(a, b) if u != v) == 1:
            polyline([a, b], m_box, 0.014)

M = {"qc": mat("qc", "#1F5F99", coat=0.4, subsurf=0.15), "border": mat("border", "#C3C9D0", coat=0.3),
     "ghost": mat("ghost", "#C3C9D0", alpha=0.25), "disc": mat("disc", "#7A838C", coat=0.3),
     "thin": mat("thin", "#CC79A7", coat=0.3, subsurf=0.1)}
R = {"qc": mat("qc_r", "#7FC0FF", emit=2.2), "border": mat("border_r", "#F2F5F8", emit=1.6),
     "disc": mat("disc_r", "#C9D1DB", emit=1.4), "thin": mat("thin_r", "#F7A8D6", emit=1.8)}

objs = []
# 1 QC-passing: one piece, inside the crop, through many sections
qc = tube("QC", [(-1.45, -1.1, -1.0), (-1.0, -0.85, -0.35), (-0.8, -0.45, 0.3), (-0.9, 0.0, 0.85)], 0.25)
qc.data.materials.append(M["qc"]); objs.append((qc, R["qc"]))
# 2 Border-truncated: leaves the crop through the +x face (inside part solid, outside part faint)
bt = tube("Border", [(0.95, 0.85, -0.6), (1.6, 1.1, -0.38), (2.7, 1.35, -0.15)], 0.26)
cutter = box_obj("cutter", (3, 8, 8), (HX + 1.5, 0, 0)); cutter.hide_render = True
ghost = bt.copy(); ghost.data = bt.data.copy(); bpy.context.collection.objects.link(ghost)
md = bt.modifiers.new("cut", "BOOLEAN"); md.operation = "DIFFERENCE"; md.object = cutter
gm = ghost.modifiers.new("keep", "BOOLEAN"); gm.operation = "INTERSECT"; gm.object = cutter
bt.data.materials.append(M["border"]); ghost.data.materials.clear(); ghost.data.materials.append(M["ghost"])
objs.append((bt, R["border"]))
# 3 Disconnected: one label on two separated pieces
d1 = metaball("DiscA", [("ELLIPSOID", Vector((0.15, -1.35, 0.42)), 0.3, 1.35, 1.0, 1.0)])
d2 = metaball("DiscB", [("ELLIPSOID", Vector((0.95, -1.2, 0.66)), 0.28, 1.1, 1.0, 1.0)])
for d in (d1, d2):
    d.data.materials.append(M["disc"]); objs.append((d, R["disc"]))
# 4 Fewer than three sections: flat, long in-plane
th = metaball("Thin", [("ELLIPSOID", Vector((0.55, 0.3, SECTIONS[1])), 0.25, 4.2, 1.1, 0.3)])
th.rotation_euler = (0, 0, math.radians(28)); bpy.context.view_layer.update()
th.data.materials.append(M["thin"]); objs.append((th, R["thin"]))
# 5 Split across sections: one object carried by two labels, separated at one section
s1 = tube("SplitA", [(-1.35, 1.35, -1.0), (-1.3, 1.4, -0.3)], 0.22)
s2 = tube("SplitB", [(-1.29, 1.41, -0.06), (-1.25, 1.45, 0.75)], 0.22)
for s in (s1, s2):
    s.data.materials.append(M["thin"]); objs.append((s, R["thin"]))
bpy.context.view_layer.update()
for o, rm in objs:
    rings(o, rm)

# soft shadow on an invisible floor
# (no floor: a shadow catcher greys the light variant)

for name, loc, energy, size in (("key", (-4, -6, 9), 1100, 6), ("fill", (7, -3, 3), 320, 8), ("rim", (2, 7, 6), 300, 6)):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size
    lo = bpy.data.objects.new(name, ld); lo.location = loc
    lo.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(lo)

cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = 7.9
cam = bpy.data.objects.new("cam", cd); bpy.context.collection.objects.link(cam)
cam.location = (6.2, -8.4, 5.4)
cam.rotation_euler = (Vector((0.25, 0.15, -0.15)) - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.camera = cam
bpy.context.view_layer.update()

anchors = {"1": (-1.15, -0.95, 0.15), "2": (1.55, 1.1, -0.1), "3a": (0.15, -1.35, 0.3), "3b": (0.95, -1.2, 0.55),
           "4": (1.05, 0.62, SECTIONS[1]), "5": (-1.27, 1.43, 0.95)}
px = {}
for k, co in anchors.items():
    v = world_to_camera_view(sc, cam, Vector(co))
    px[k] = (round(v.x * RES[0], 1), round((1 - v.y) * RES[1], 1))
(OUT / "checks_anchors.json").write_text(json.dumps({"resolution": RES, "anchors_px": px}, indent=1))
if "--anchors" in sys.argv:
    print("anchors", px); raise SystemExit(0)
sc.render.filepath = str(OUT / "checks_render.png"); sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
bpy.ops.render.render(write_still=True)
print("rendered", px)
