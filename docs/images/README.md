# README figures

Schematics for the project README. They contain no specimen data.

| Figure | Made with | Script |
|---|---|---|
| `mitocensus_banner.png` | Blender (Cycles) and Matplotlib | `src/blender_banner.py`, `src/compose_banner.py` |
| `mitocensus_object_checks_{light,dark}.png` | Blender (Cycles) and Matplotlib | `src/blender_checks_v2.py`, `src/compose_checks_v2.py` |
| `mitocensus_workflow_{light,dark}.{png,svg}` | Matplotlib | `src/make_workflow_cards.py` |

To rebuild:

```bash
blender -b --factory-startup -P src/blender_banner.py -- OUTDIR
python src/compose_banner.py OUTDIR OUTDIR
blender -b --factory-startup -P src/blender_checks_v2.py -- OUTDIR
python src/compose_checks_v2.py OUTDIR OUTDIR
python src/make_workflow_cards.py OUTDIR
```
