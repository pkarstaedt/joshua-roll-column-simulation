# Joshua Roll Column Simulation

Procedural 3D reconstruction workflow for wrapping the [Joshua Roll](https://en.wikipedia.org/wiki/Joshua_Roll) imagery onto a triumphal column, as was originally intended.

## Intro 
- 10m long, 30cm, 10th century, 
- joshua's campaign in the holy land
- heraclius campaign in the last great war, 603 to 
- triumphal column, 10.5 - 14 meters
- reading book, chapter 10.5
- very intriguing
- wanted to try out
`[Write a short paragraph here about why you decided to build this project.]`


## Provenance and Source Notes
- The Joshua Roll source images in this project come from the [Vatican Library](https://digi.vatlib.it/view/MSS_Pal.gr.431.pt.B).
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

## Walkable Viewer Variant
- Alternate entry point: `joshua_roll_render_walking.py`
- Purpose:
  - uses the same generation pipeline as `joshua_roll_render.py`,
  - opens a walkable first-person viewer with a ground plane,
  - supports camera-relative movement (`WASD`) and vertical flying via `Space`,
  - keeps export behavior (`E`) consistent with the main script.

## Column-Focused Variant
- Alternate entry point: `joshua_roll_render_column.py`
- Purpose:
  - uses the same core pipeline,
  - is configured to look more like a conventional architectural column,
  - serves as an alternative to the triumphal-column-oriented defaults.

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
- `joshua_roll_render_walking.py`: walkable first-person viewer variant of the pipeline.
- `joshua_roll_render_column.py`: alternative column-focused configuration/view.
- `README.md`: project documentation.
- `jc_images/`: source image slices.
- Root working assets:
  - `jc_roll_small.jpg`,
  - `jc_roll_small_depth.png`,
  - `plinth_side.png`,
  - `plinth_side_depth.png`,
  - `theodosius.glb`.
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
2. Edit the config block at the top of either script:
- `joshua_roll_render.py` for orbit-style viewing
- `joshua_roll_render_walking.py` for walkable viewing
  - `joshua_roll_render_column.py` for a more conventional column-oriented setup
3. Run one of:

```bash
python joshua_roll_render.py
python joshua_roll_render_walking.py
python joshua_roll_render_column.py
```

## Viewer Controls (`joshua_roll_render.py`)
- Left-drag: orbit
- Right-drag: pan
- Scroll: zoom
- `R`: reset camera
- `A`: toggle auto-rotate
- `E`: export using current `EXPORT` setting
- `Q` or `Esc`: quit

## Walkable Controls (`joshua_roll_render_walking.py`)
- Mouse move: look around
- `W A S D`: camera-relative move
- Hold `Space`: fly upward
- Double-tap `Space`: drop back to ground height
- `Tab`: toggle mouse capture
- `R`: reset camera
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
- `joshua_roll_render_walking.py` shares most config behavior with the main script, but uses a first-person walkable viewer.
