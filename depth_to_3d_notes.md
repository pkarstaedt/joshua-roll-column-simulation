# depth_to_3d.py — How It Works & Development Notes

## Overview

`depth_to_3d.py` takes a flat colour image and a matching depth map and turns
them into an interactive 3-D relief mesh. The mesh can be previewed in a
real-time OpenGL window (pyglet) and exported to formats that Blender can
import directly.

---

## Pipeline

```
colour image  ──┐
                ├──► load_and_preprocess()
depth map     ──┘         │
                          ▼
                     build_mesh()
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         run_viewer()           export_obj()
         (pyglet / OpenGL)      export_glb()
                                (on E keypress)
```

---

## Step-by-step breakdown

### 1. `load_and_preprocess()`

| What happens | Why |
|---|---|
| Colour image is opened and forced to RGB | Strips alpha channels and normalises colour mode |
| Depth map is opened and converted to greyscale (`"L"`) | Only luminance matters for displacement; colour depth maps are reduced to a single channel |
| Depth map is **resized to match the colour image** using Lanczos resampling | Depth maps are often generated at a different resolution than the source image (e.g. by a neural network); they must align pixel-for-pixel before meshing |
| **Contrast** is applied to the depth map via `ImageEnhance.Contrast` | Depth maps from AI models (e.g. MiDaS, Depth Anything) are often low-contrast and "flat". Boosting contrast (try 1.5–3.0) separates near and far regions more clearly |
| Both images are **downsampled** to the mesh resolution (`W // DOWNSAMPLE`, `H // DOWNSAMPLE`) | A 4K image has ~8 million pixels; a mesh with 8M vertices is impractical. Downsampling by 4 gives a 1M-vertex mesh that is still very detailed |
| Depth values are normalised to `[0.0, 1.0]` (float32) | Keeps the displacement maths independent of bit depth |

**Key learning:** Lanczos resampling is important here — nearest-neighbour or
bilinear resampling of a depth map introduces staircase artefacts in the mesh
surface. Lanczos preserves smooth gradients.

---

### 2. `build_mesh()`

The mesh is a **displaced plane** — a regular grid of vertices where each
vertex's Z coordinate is pushed forward by the depth value at that pixel.

```
vertex position:
  x = linspace(-aspect, +aspect, W)   ← preserves image aspect ratio
  y = linspace(+1, -1, H)             ← Y is flipped so row 0 is at the top
  z = depth_value × DEPTH_SCALE
```

**Quad triangulation:**  
Each 2×2 block of grid vertices forms two triangles (a quad split along the
diagonal). For a grid of `H × W` vertices there are `(H-1) × (W-1) × 2`
triangles.

```
tl──tr        tl──tr   tl
│  /│    →    │ /    \ │
│ / │         │/      \│
bl──br        bl        br
```

**Per-vertex normals** are computed by accumulating the cross-product face
normals of all triangles that share each vertex, then normalising. This gives
smooth shading across the surface.

**Key learning:** Using `np.add.at()` for normal accumulation is correct but
slow for large meshes (it is not vectorised). For full-resolution 4K meshes
this is the main bottleneck. A faster approach would use `np.bincount` or
sparse matrix operations, but for typical downsampled meshes it is fast enough.

---

### 3. `export_obj()` — Blender OBJ export

Writes two files:

- **`<stem>.obj`** — vertex positions (`v`), UV coordinates (`vt`), normals
  (`vn`), and triangle faces (`f`). OBJ indices are 1-based.
- **`<stem>.mtl`** — material definition that references the original colour
  image as a diffuse texture (`map_Kd`).

**Important:** The `.obj`, `.mtl`, and the texture image must all be in the
**same folder** when importing into Blender. Blender resolves the texture path
relative to the MTL file.

To import in Blender:  
`File → Import → Wavefront (.obj)` → select the `.obj` file.

**Key learning:** OBJ face entries use the format `v/vt/vn` (vertex / UV /
normal index). Since our mesh has one UV and one normal per vertex (they share
the same index), the face lines simplify to `i/i/i` for each corner — no need
for a separate index buffer per attribute.

---

### 4. `export_glb()` — Blender GLB export

Uses the `trimesh` library to build a mesh object, attach PBR material with
the texture embedded, and write a self-contained binary GLTF (`.glb`) file.

GLB embeds the texture inside the file itself, so there is no separate image
file to manage.

To import in Blender:  
`File → Import → glTF 2.0 (.glb/.gltf)` → select the `.glb` file.

Requires: `pip install trimesh[easy]`

---

### 5. `run_viewer()` — pyglet / OpenGL viewer

Uses pyglet's low-level OpenGL bindings (fixed-function pipeline, no custom
shaders required) to render the mesh in real time.

**Rendering path:**
1. Perspective projection matrix is built manually and loaded via `glLoadMatrixf`
2. Camera transform (translate + rotate) is applied via `glTranslatef` / `glRotatef`
3. A single directional light (`GL_LIGHT0`) provides diffuse + ambient shading
4. The colour image is uploaded as a mipmapped OpenGL texture
5. Vertex, normal, and UV arrays are passed to OpenGL via client-state pointers
6. `glDrawElements` draws all triangles in a single call

**Camera state** is stored in a plain dict (`cam`) that is mutated by mouse
event handlers — a simple alternative to a camera class.

**Key learning:** pyglet ≥ 2.0 changed several APIs. The `GL_UNSIGNED_INT`
index type requires that index arrays are `uint32`, not `int32` — passing
signed integers silently produces garbage geometry. The `glGenerateMipmap`
call must happen *after* `glTexImage2D`.

---

## Configuration variables

All settings live at the top of the script in the `CONFIG` block:

| Variable | Type | Description |
|---|---|---|
| `IMAGE` | `str` | Path to the source colour image |
| `DEPTH` | `str` | Path to the depth map (any format Pillow can read) |
| `CONTRAST` | `float` | Contrast multiplier for the depth map. `1.0` = unchanged. Try `1.5`–`3.0` for AI-generated depth maps |
| `DEPTH_SCALE` | `float` | Z-displacement intensity. `0.0` = flat, `1.0` = maximum relief. `0.2`–`0.5` is a good starting range |
| `DOWNSAMPLE` | `int` | Mesh resolution divisor. `1` = one vertex per pixel (slow). `4` = default. `8` = fast preview |
| `EXPORT` | `str` | Export format when `E` is pressed: `"obj"`, `"glb"`, or `"none"` |
| `OUT` | `str` | Output filename stem (no extension). Defaults to `<image_name>_output` |

---

## Viewer controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit (yaw / pitch) |
| Right mouse drag | Pan |
| Scroll wheel | Zoom in / out |
| `E` | Export mesh to file (OBJ or GLB per `EXPORT` setting) |
| `R` | Reset camera to default position |
| `Q` or `Esc` | Close viewer and exit |

---

## Dependencies

```
Pillow>=10.0.0      # image loading, resize, contrast
numpy>=1.24.0       # mesh arrays
pyglet>=2.0.0       # OpenGL viewer window
trimesh[easy]       # GLB export only (optional)
```

Install: `pip install -r requirements_depth3d.txt`

---

## Learnings & gotchas

### Depth map quality matters most
The quality of the final mesh is almost entirely determined by the depth map.
AI-generated depth maps (MiDaS, Depth Anything, ZoeDepth, etc.) produce
relative depth — they are not metrically accurate, but they are good enough
for artistic 3-D relief. The `CONTRAST` parameter is the most impactful
single knob for improving the result.

### Downsample before meshing, not after
It is tempting to build the full-resolution mesh and then simplify it. In
practice, downsampling the *images* before meshing is far faster and produces
cleaner results because Lanczos resampling anti-aliases the depth gradient
before it becomes geometry.

### Aspect ratio in the mesh coordinate system
The mesh X range is `[-aspect, +aspect]` (where `aspect = W / H`) rather than
`[-1, +1]`. This means the mesh has the correct proportions in world space
regardless of image dimensions, and the camera does not need to be adjusted
per image.

### OBJ is the most reliable Blender import format
GLB is more modern and self-contained, but the `trimesh` PBR material path
has occasional compatibility issues with older Blender versions. OBJ + MTL
is universally supported and the texture assignment is transparent and easy
to debug.

### Fixed-function OpenGL is sufficient here
Using `GL_LIGHTING`, `GL_COLOR_MATERIAL`, and client-state arrays avoids the
need for GLSL shaders, which simplifies the code considerably. The trade-off
is that the rendering is less flexible (no PBR, no shadows), but for a
preview tool this is perfectly adequate.

### `np.add.at` is the bottleneck for large meshes
Normal computation with `np.add.at` is O(triangles) but not vectorised. For a
`DOWNSAMPLE=1` run on a 4K image (~8M triangles) this takes tens of seconds.
If full-resolution meshes are needed regularly, replacing this with a
`scipy.sparse` matrix multiply or a `numba`-jitted loop would give a large
speedup.
