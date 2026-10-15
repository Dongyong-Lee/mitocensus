"""README banner: mitochondria passing through serial sections, with their cross-section slices (synthetic shapes).

Run: Blender -b --factory-startup -P blender_banner.py -- OUTDIR   ->  OUTDIR/banner_render.png (1600 x 560, opaque)"""
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

OUT = Path(sys.argv[sys.argv.index("--") + 1])
OUT.mkdir(parents=True, exist_ok=True)
RES = (1600, 560)
random.seed(7)
PLANES_Z = [-0.9, -0.54, -0.18, 0.18, 0.54, 0.9]


def lin(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c) + (1.0,)


def mat(name, hexcol, rough=0.35, alpha=1.0, emit=0.0, subsurf=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin(hexcol)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Alpha"].default_value = alpha
    if emit:
        b.inputs["Emission Color"].default_value = lin(hexcol)
        b.inputs["Emission Strength"].default_value = emit
    if subsurf:
        b.inputs["Subsurface Weight"].default_value = subsurf
        b.inputs["Subsurface Radius"].default_value = (0.3, 0.3, 0.3)
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    return m


def metaball_mesh(name, pts, radius, n=20, res=0.05):
    mb = bpy.data.metaballs.new(name)
    mb.resolution, mb.render_resolution = res, res / 2
    o = bpy.data.objects.new(name, mb)
    bpy.context.collection.objects.link(o)
    for i in range(len(pts) - 1):
        a, b = Vector(pts[i]), Vector(pts[i + 1])
        for k in range(n):
            e = mb.elements.new(type="BALL")
            e.co = a.lerp(b, k / n)
            e.radius = radius * (0.92 + 0.16 * math.sin(3.1 * (i + k / n)))
    e = mb.elements.new(type="BALL"); e.co = Vector(pts[-1]); e.radius = radius
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.convert(target="MESH")
    m = bpy.context.view_layer.objects.active
    bpy.ops.object.shade_smooth()
    return m


def slab(z, thick=0.022, size=(14, 10)):
    me = bpy.data.meshes.new("slab")
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("slab", me)
    bpy.context.collection.objects.link(o)
    o.scale = (size[0], size[1], thick)
    o.location = (0, 0, z)
    return o


def slices(src, color_mat):
    """Thin cross-section discs where the object meets each section plane."""
    dg = bpy.context.evaluated_depsgraph_get()
    for z in PLANES_Z:
        cut = slab(z)
        tmp = src.copy(); tmp.data = src.data.copy()
        bpy.context.collection.objects.link(tmp)
        mod = tmp.modifiers.new("x", "BOOLEAN"); mod.operation = "INTERSECT"; mod.object = cut; mod.solver = "EXACT"
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        bpy.data.objects.remove(tmp); bpy.data.objects.remove(cut)
        if len(me.vertices):
            s = bpy.data.objects.new("slice", me)
            bpy.context.collection.objects.link(s)
            s.data.materials.clear(); s.data.materials.append(color_mat)
            s.scale = (1.012, 1.012, 1.0)


# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 160
sc.cycles.use_denoising = True
sc.render.film_transparent = False
sc.render.resolution_x, sc.render.resolution_y = RES
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = lin("#0A1120"); bg.inputs["Strength"].default_value = 1.0

# serial sections: faint glass-like sheets with a fine edge
m_sheet = mat("sheet", "#9FB7D6", rough=0.25, alpha=0.05)
m_edge = mat("edge", "#6F8FB8", rough=0.6, emit=0.55)
for z in PLANES_Z:
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0.3, 0.15, z))
    p = bpy.context.active_object
    p.scale = (5.6, 3.3, 1)
    p.data.materials.append(m_sheet)
    cu = bpy.data.curves.new("edge", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.006
    sp = cu.splines.new("POLY"); pts = [(-2.5, -1.5), (3.1, -1.5), (3.1, 1.8), (-2.5, 1.8), (-2.5, -1.5)]
    sp.points.add(len(pts) - 1)
    for q, (x, y) in zip(sp.points, pts):
        q.co = (x, y, z, 1)
    eo = bpy.data.objects.new("edge", cu); bpy.context.collection.objects.link(eo); cu.materials.append(m_edge)

# mitochondria (synthetic): QC-passing tubes in blue, one thin label in pink, one cut by the crop face in grey
blue_body = mat("blue", "#2F7FD0", rough=0.32, subsurf=0.25, coat=0.4)
blue_body2 = mat("blue2", "#3F9BE8", rough=0.32, subsurf=0.25, coat=0.4)
blue_slice = mat("blue_slice", "#9CD3FF", emit=6.0)
pink_body = mat("pink", "#E07BB5", rough=0.35, subsurf=0.2, coat=0.3)
pink_slice = mat("pink_slice", "#FFC2E6", emit=6.0)
grey_body = mat("grey", "#C9D3E0", rough=0.4, coat=0.3)
grey_slice = mat("grey_slice", "#F2F6FB", emit=4.0)

tubes = [
    ([(-1.9, -0.9, -1.05), (-1.5, -0.6, -0.3), (-1.65, -0.2, 0.5), (-1.3, 0.1, 1.05)], 0.21, blue_body),
    ([(-0.7, -1.0, -1.1), (-0.35, -0.75, -0.35), (-0.05, -0.8, 0.35), (0.3, -0.5, 1.0)], 0.23, blue_body2),
    ([(0.6, 0.9, -1.05), (0.95, 0.6, -0.25), (0.8, 0.3, 0.55), (1.15, 0.05, 1.08)], 0.2, blue_body),
    ([(1.7, -1.0, -1.0), (1.45, -0.55, -0.1), (1.85, -0.15, 0.7)], 0.24, blue_body2),
    ([(-0.9, 1.0, -0.6), (-0.6, 1.25, 0.15), (-0.8, 1.45, 0.95)], 0.19, blue_body),
    ([(2.4, 0.9, -0.75), (2.65, 1.2, 0.1), (2.45, 1.5, 0.85)], 0.2, blue_body2),
    ([(0.9, -1.15, 0.0), (1.2, -0.95, 0.75)], 0.18, blue_body),
]
objs = []
for k, (pts, r, m) in enumerate(tubes):
    o = metaball_mesh(f"Mito{k}", pts, r)
    o.data.materials.append(m)
    objs.append((o, blue_slice))
# thin label: flat, in one section, long in-plane
thin = metaball_mesh("Thin", [(0.0, 0.35, 0.18), (1.05, 0.65, 0.18)], 0.16, n=24)
thin.scale = (1.0, 1.0, 0.35); thin.data.materials.append(pink_body)
bpy.context.view_layer.update()
objs.append((thin, pink_slice))

bpy.context.view_layer.update()
for o, sm in objs:
    slices(o, sm)

# lights
for name, loc, energy, size, col in (("key", (-3, -6, 7), 1400, 5, "#F4F8FF"), ("rim", (5, 5, 3), 900, 4, "#6FA8FF"),
                                     ("fill", (6, -5, -2), 260, 6, "#FFE8D6")):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size; ld.color = lin(col)[:3]
    lo = bpy.data.objects.new(name, ld); lo.location = loc
    lo.rotation_euler = (Vector((0.5, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(lo)

# camera: objects sit in the right two thirds, the left third stays dark for the title
cd = bpy.data.cameras.new("cam"); cd.lens = 43
cd.dof.use_dof = True; cd.dof.aperture_fstop = 3.2
cam = bpy.data.objects.new("cam", cd); bpy.context.collection.objects.link(cam)
cam.location = (-0.9, -12.0, 4.3)
target = Vector((0.35, 0.2, -0.1))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cd.dof.focus_distance = (Vector((0.3, 0, 0)) - cam.location).length
cd.shift_x = -0.215
sc.camera = cam

sc.render.filepath = str(OUT / "banner_render.png")
sc.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("rendered", sc.render.filepath)
