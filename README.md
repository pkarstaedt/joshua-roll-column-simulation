# Joshua Roll Column Simulation

Procedural 3D reconstruction workflow for wrapping the Joshua Roll imagery onto a helical column, then extending it with architectural elements (base, moldings, capital) and a top statue.

## Intro (Add Your Motivation Here)
`[Write a short paragraph here about why you decided to build this project.]`

## Provenance and Source Notes
- The Joshua Roll source images in this project come from the Vatican Library.
- The source images were lightly edited by the repository author to make transitions between individual pictures less visible.
- The source material used for the image sequence was about 15 cm.
- The depth map used in this repo was created with ComfyUI and Lotus.

## Main Script
- Primary entry point: `joshua_roll_render.py`
- Purpose:
  - loads color + depth images,
  - generates relief geometry,
  - wraps it around a helical column,
  - builds optional architectural parts,
  - previews in an interactive OpenGL window,
  - exports OBJ/MTL and/or GLB.

## Core Features in `joshua_roll_render.py`
- Helical wrap of 2D image strip to cylindrical geometry.
- Depth-based relief generation with:
  - contrast,
  - threshold,
  - blur,
  - depth scale.
- Optional structural parts:
  - inner cylinder,
  - stepped base and plinth,
  - endcap,
  - moldings,
  - procedural capital.
- Optional top statue from GLB (`theodosius.glb`), auto-scaled and placed above the capital.
- Interactive viewer:
  - orbit/pan/zoom,
  - configurable camera start,
  - configurable light position and intensity.
- Export modes:
  - `obj`,
  - `glb`,
  - `both`,
  - `none`.

## Repository Layout
- `joshua_roll_render.py`: main generation/view/export pipeline.
- `README.md`: project documentation.
- `jc_images/`: source image slices.
- Root working assets:
  - `jc_roll_small.jpg`,
  - `jc_roll_small_depth.png`,
  - `plinth_side.png`,
  - `plinth_side_depth.png`,
  - `theodosius.glb`.
- `supporting_material/`: legacy scripts, experiments, and auxiliary assets.
- Notes:
  - `joshua-roll-column-notes.md`,
  - `depth_to_3d_notes.md`.

## Installation
Use Python 3.10+ (3.11/3.12 recommended).

Install required packages:

```bash
pip install numpy pillow pyglet trimesh
```

Optional extras for GLB handling in some environments:

```bash
pip install "trimesh[easy]"
```

## Quick Start
1. Put the required texture/depth/model files in the repository root (or update paths in config).
2. Edit the config block at the top of `joshua_roll_render.py`.
3. Run:

```bash
python joshua_roll_render.py
```

## Viewer Controls
- Left-drag: orbit
- Right-drag: pan
- Scroll: zoom
- `R`: reset camera
- `A`: toggle auto-rotate
- `E`: export using current `EXPORT` setting
- `Q` or `Esc`: quit

## Export
Set `EXPORT` in the config:
- `EXPORT = "obj"`: writes `OUT.obj` + `OUT.mtl`
- `EXPORT = "glb"`: writes `OUT.glb`
- `EXPORT = "both"`: writes both formats
- `EXPORT = "none"`: disables export hotkey

## Typical Workflow
1. Build/clean source strip imagery.
2. Generate depth map (ComfyUI + Lotus).
3. Tune depth and wrap settings in config.
4. Tune architectural options (base/capital/statue).
5. Preview in viewer.
6. Export for Blender or other DCC tools.

## Notes
- The script is heavily config-driven; most behavior is controlled at the top of `joshua_roll_render.py`.
- `supporting_material/` contains earlier iterations and helper scripts that may still be useful for preprocessing and debugging.
