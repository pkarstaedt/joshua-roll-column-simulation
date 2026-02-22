"""
depth_to_3d.py
==============
Creates a 3-D mesh from an image and its depth map, previews it in a
pyglet window, and exports it to OBJ/MTL (Blender-ready) or GLB.

Edit the CONFIG block below to set all options, then run:
    python depth_to_3d.py

Controls in the viewer
----------------------
  Left-drag   – orbit
  Right-drag  – pan
  Scroll      – zoom
  R           – reset camera
  Q / Escape  – quit
"""

import os
import sys
import math
import time
import ctypes

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these values
# ──────────────────────────────────────────────────────────────────────────────

# Source colour image
IMAGE = "jc_roll_small.jpg"

# Depth map image (greyscale or colour)
DEPTH = "jc_roll_small_depth.png"

# Contrast multiplier applied to the depth map before meshing.
# 1.0 = no change.  Try 1.5–3.0 to sharpen a flat-looking depth map.
CONTRAST = 1.5

# Depth threshold (0.0 – 1.0).
# Any depth value below this level is clamped to 0 (stays flat).
# 0.0 = disabled (all depths raised).  Try 0.1–0.3 to suppress background noise.
DEPTH_THRESHOLD = 0.4

# Blur radius applied to the depth map after thresholding (in pixels, pre-downsample).
# Softens the transitions between raised and flat areas.
# 0 = disabled.  Try 2–10 for gentle feathering, 20–50 for very soft edges.
BLUR_RADIUS = 4

# Z-displacement intensity in world units.
# In column mode (WRAP_COLUMN=True), R=1.0 by default, so:
#   0.05 → ~5% protrusion  (subtle bas-relief)
#   0.10 → ~10% protrusion (strong relief, like Trajan's Column)
# In flat mode these are arbitrary scene units (0.2–0.5 works well).
DEPTH_SCALE = 0.07

# Thickness of the solid backing plate in world units.
# In column mode this is the shell thickness (inward from the surface).
# 0.0 = no backing.  ~0.02–0.05 = thin shell.
BACKING_DEPTH = 0.02

# Mesh resolution divisor.
# 1 = one vertex per pixel (full res, very slow for large images).
# 4 = default (good balance of quality and speed).
# 8 = fast preview.
DOWNSAMPLE = 4

# ── Column / helical wrap ─────────────────────────────────────────────────────
# Set WRAP_COLUMN = True to bend the relief strip into a triumphal column.
# When False the mesh is a flat plaque (original behaviour).
WRAP_COLUMN = True

# Helix angle in degrees.  15° matches the classic Trajan's Column geometry:
# the strip rises at 15° so the image snakes upward as it wraps around.
HELIX_ANGLE_DEG = 17.0

# Column radius in world units.
# The mesh always wraps exactly once around the cylinder (perfect seam).
# R controls the physical size of the column relative to the depth displacement.
# DEPTH_SCALE is also in world units, so relief protrudes by DEPTH_SCALE/R of
# the column radius — keep DEPTH_SCALE << R for realistic bas-relief.
# None = default R = 1.0  (diameter = 2.0, depth_scale = 0.05 → ~2.5% protrusion)
COLUMN_RADIUS = None   # None = 1.0

# ── Column parts ─────────────────────────────────────────────────────────────
# Each part can be toggled independently.  All use the same COLUMN_RADIUS and
# HELIX_ANGLE_DEG so they fit together automatically.

# Fill the interior of the helix with a solid cylinder.
# Height spans from Z=0 (column base) to Z=col_height (top of helix).
ADD_INNER_CYLINDER = True
# Number of sides on the inner cylinder (more = smoother).
INNER_CYLINDER_SEGMENTS = 64

# Add a rectangular base/plinth below the column.
ADD_BASE = True
# Base style: "stepped" (monument plinth) or "block" (single box).
BASE_STYLE = "stepped"
# How much the base extends beyond the column radius on each side (world units).
BASE_OVERHANG = 1
# Height of the base block (world units).
BASE_HEIGHT = 6
# Top plinth height directly under the column (used by stepped style).
BASE_TOP_PLINTH_HEIGHT = 4
# Number of stepped tiers under the top plinth (used by stepped style).
BASE_STEPS = 6
# Horizontal expansion per step (world units, each side).
BASE_STEP_RUN = 0.35
# Step height (world units). None = auto from BASE_HEIGHT.
BASE_STEP_HEIGHT = None
# Smaller stepped tiers above the top plinth (just under the column).
BASE_TOP_STEPS = 3
# Horizontal expansion per top step (world units, each side).
BASE_TOP_STEP_RUN = 0.2
# Height of each top step (world units).
BASE_TOP_STEP_HEIGHT = 0.3

# Optional dedicated texture/depth for the top plinth side band.
PLINTH_SIDE_IMAGE = "plinth_side.png"
PLINTH_SIDE_DEPTH = "plinth_side_depth.png"
# Outward relief depth on the top plinth side (world units).
PLINTH_SIDE_DEPTH_SCALE = 0.4
# Tessellation for plinth side displacement mapping.
PLINTH_SIDE_U_SEGMENTS = 160
PLINTH_SIDE_V_SEGMENTS = 24
# Flip plinth side texture/depth vertically on each face.
PLINTH_SIDE_FLIP_V = True
# Invert plinth depth luminance before displacement (use if relief looks inside-out).
PLINTH_SIDE_DEPTH_INVERT = False
# Blur radius for plinth step texturing (keeps top plinth side detail separate).
PLINTH_STEP_TEXTURE_BLUR_RADIUS = 18.0
# Recess the plinth side band inward so corners stay closed/solid.
PLINTH_SIDE_RECESS = 0.02
# Fade displacement to zero near vertical edges of each face.
PLINTH_CORNER_BLEND_FRACTION = 0.08

# Add a cylindrical endcap (drum) on top of the column.
ADD_ENDCAP = True
# Radius of the endcap (world units).  None = same as COLUMN_RADIUS.
ENDCAP_RADIUS = 1.3   # None = COLUMN_RADIUS
# Height of the endcap drum (world units).
ENDCAP_HEIGHT = 0.2
# Number of sides on the endcap.
ENDCAP_SEGMENTS = 64

# Add a classical-style capital (flared + decorated) on top of the column.
ADD_CAPITAL = True
CAPITAL_HEIGHT = 1.7
CAPITAL_RADIUS_MULT = 1.65
CAPITAL_SEGMENTS = 96
CAPITAL_WAVE_COUNT = 20
CAPITAL_WAVE_AMPLITUDE = 0.06
CAPITAL_ABACUS_OVERHANG = 0.05
CAPITAL_ABACUS_HEIGHT = 0.32
# Optional dedicated texture for capital sides.
CAPITAL_TEXTURE_IMAGE = "plinth_side.png"
CAPITAL_TEXTURE_FLIP_V = False
CAPITAL_TEXTURE_BLUR_RADIUS = 20.0
CAPITAL_TEXTURE_WRAP_PER_SIDE = False

# Optional external GLB statue placed above the capital.
ADD_TOP_STATUE = True
TOP_STATUE_GLB = "theodosius.glb"
# Statue height as a multiple of CAPITAL_HEIGHT.
TOP_STATUE_HEIGHT_MULT = 2.9
# Extra vertical gap above capital top.
TOP_STATUE_Z_GAP = 0.0
# GLB texture mapping tweaks.
TOP_STATUE_TEXTURE_FLIP_V = True

# Clip wrapped helix triangles at the top cap plane (Z <= col_height).
CLIP_HELIX_AT_ENDCAP = True
HELIX_TOP_CLIP_EPS = 1e-4

# Add an integrated divider lip under the helical image strip.
# This is generated inside the wrapped strip mesh, not as a separate part.
ADD_LIP = True
# Lip vertical height in world units. None = auto (base column height / 20).
LIP_HEIGHT = 0.05
# Lip radial protrusion in world units.
LIP_DEPTH = 0.12
# Fraction of texture V-range used for lip texturing.
# 0.03 means lip maps to the bottom 3% of the texture.
LIP_TEXTURE_STRIP_FRACTION = 0.03
# Heavy blur radius (pixels) applied only to the lip texture strip rendering.
LIP_TEXTURE_BLUR_RADIUS = 30.0

# Smoothly taper relief near strip start/end to reduce hard transitions.
END_FADE_ENABLED = True
# Fraction of strip length to fade at each end (0.03 = 3% on left + 3% on right).
END_FADE_FRACTION = 0.03
# Fade curve exponent. 1.0=linear, >1 sharper near center, <1 softer.
END_FADE_POWER = 1.0

# Mirror interior content into strip ends (start/end) to avoid sparse edge areas.
# This keeps strip width unchanged, so column/base/endcap placement stays unchanged.
MIRROR_EDGE_ENABLED = True
MIRROR_EDGE_FRACTION = 0.10

# Add decorative torus/fillet moldings at column bottom/top.
ADD_MOLDINGS = True
ADD_BOTTOM_MOLDING = True
ADD_TOP_MOLDING = True
# Torus center radius = COLUMN_RADIUS + offset.
MOLDING_RADIUS_OFFSET = 0.08
# Torus tube radius (thickness of molding profile).
MOLDING_TUBE_RADIUS = 0.1
# Torus tessellation.
MOLDING_MAJOR_SEGMENTS = 96
MOLDING_TUBE_SEGMENTS = 24
# Vertical placement.
MOLDING_BOTTOM_Z = 0.0
MOLDING_TOP_Z_OFFSET = 0.0

# Viewer startup (column mode)
# Front-on view of the column: around -90 on pitch shows full height side-on.
VIEW_COLUMN_INIT_YAW = 0.0
VIEW_COLUMN_INIT_PITCH = -90.0
# Camera distance as a multiple of computed column height.
VIEW_COLUMN_DIST_FACTOR = 2.5
# Center camera vertically on the column at startup.
VIEW_COLUMN_CENTER_HEIGHT = True

# Lighting in viewer (OpenGL positional light):
# (x, y, z, w) where w=1.0 is positional, w=0.0 is directional.
LIGHT_POSITION = (2.0, 3.0, 4.0, 1.0)
LIGHT_DIFFUSE = (1.0, 1.0, 1.0, 1.0)
LIGHT_AMBIENT = (0.45, 0.45, 0.45, 1.0)

# Auto-derive stone colors for non-textured parts from IMAGE.
AUTO_MATCH_PART_COLORS = True

# ── Export ────────────────────────────────────────────────────────────────────
# Export format triggered by pressing E in the viewer:
#   "obj"  – writes <OUT>.obj + <OUT>.mtl  (import in Blender via
#             File → Import → Wavefront OBJ; keep the image file in the same folder).
#   "glb"  – single binary GLTF file (requires: pip install trimesh[easy]).
#             Import in Blender via File → Import → glTF 2.0.
#   "both" – writes both OBJ/MTL and GLB in one keypress.
#   "none" – disable export entirely (E key does nothing).
EXPORT = "obj"

# Output filename stem (no extension).
OUT = IMAGE.split(".")[0]+"_output"

# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Image / depth-map loading & preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str, depth_path: str,
                        contrast: float, downsample: int,
                        depth_threshold: float = 0.0,
                        blur_radius: float = 0):
    """
    Returns
    -------
    colour_small : np.ndarray  uint8  (H, W, 3)
    depth_norm   : np.ndarray  float32 (H, W)  values in [0, 1]
    """
    from PIL import ImageFilter

    colour = Image.open(image_path).convert("RGB")
    W, H   = colour.size
    print(f"[info] Source image  : {W} × {H} px")

    # ── depth map ──────────────────────────────────────────────────────────
    depth_img = Image.open(depth_path).convert("L")          # greyscale
    dW, dH    = depth_img.size
    print(f"[info] Depth map     : {dW} × {dH} px")

    # Resize depth map to match source image if needed
    if (dW, dH) != (W, H):
        print(f"[info] Resizing depth map → {W} × {H}")
        depth_img = depth_img.resize((W, H), Image.LANCZOS)

    # Contrast adjustment
    if contrast != 1.0:
        print(f"[info] Applying depth contrast × {contrast}")
        depth_img = ImageEnhance.Contrast(depth_img).enhance(contrast)

    # Threshold — zero out depths below the cutoff
    if depth_threshold > 0.0:
        print(f"[info] Applying depth threshold  ≥ {depth_threshold:.3f}")
        d = np.array(depth_img, dtype=np.float32) / 255.0
        d[d < depth_threshold] = 0.0
        depth_img = Image.fromarray((d * 255).astype(np.uint8), mode="L")

    # Blur — soften transitions between raised and flat areas
    if blur_radius > 0:
        print(f"[info] Blurring depth map  radius={blur_radius}px")
        depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # ── downsample both to mesh resolution ─────────────────────────────────
    mW = max(1, W // downsample)
    mH = max(1, H // downsample)
    print(f"[info] Mesh grid     : {mW} × {mH} = {mW*mH:,} vertices")

    colour_small = np.array(colour.resize((mW, mH), Image.LANCZOS), dtype=np.uint8)
    depth_small  = np.array(depth_img.resize((mW, mH), Image.LANCZOS), dtype=np.float32) / 255.0

    return colour_small, depth_small


def derive_part_palette_from_texture(image_path: str):
    """
    Estimate neutral stone-like colors from the texture image.
    Returns (base_rgb, lip_rgb, accent_rgb), each as uint8 tuple.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h = arr.shape[0]

    # Prioritize lower-column tones by sampling the bottom region first.
    # If this region is too small/noisy, we fall back to full image.
    bottom_start = int(round(h * 0.70))
    bottom = arr[bottom_start:, :, :]
    flat = bottom.reshape(-1, 3)

    cmax = np.max(flat, axis=1)
    cmin = np.min(flat, axis=1)
    sat = np.where(cmax > 1e-6, (cmax - cmin) / cmax, 0.0)

    # Favor neutral/mid-bright regions (typically stone tones).
    mask = (sat < 0.35) & (cmax > 0.15) & (cmax < 0.95)
    min_keep = max(256, flat.shape[0] // 200)
    if np.count_nonzero(mask) < min_keep:
        # Fallback to full image if bottom-only neutral sample is insufficient.
        flat_all = arr.reshape(-1, 3)
        cmax_all = np.max(flat_all, axis=1)
        cmin_all = np.min(flat_all, axis=1)
        sat_all = np.where(cmax_all > 1e-6, (cmax_all - cmin_all) / cmax_all, 0.0)
        mask_all = (sat_all < 0.35) & (cmax_all > 0.15) & (cmax_all < 0.95)
        if np.count_nonzero(mask_all) < max(256, flat_all.shape[0] // 200):
            sample = flat_all
        else:
            sample = flat_all[mask_all]
    else:
        sample = flat[mask]

    base = np.median(sample, axis=0)
    lip = np.clip(base * 0.88, 0.0, 1.0)
    accent = np.clip(base * 0.95, 0.0, 1.0)

    def to_u8(v):
        t = tuple(np.round(v * 255.0).astype(np.uint8).tolist())
        return (int(t[0]), int(t[1]), int(t[2]))

    return to_u8(base), to_u8(lip), to_u8(accent)


def average_rgb_from_image(image_path: str) -> tuple:
    """Return simple mean RGB as (r, g, b) uint8 tuple."""
    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    mean = np.clip(np.mean(arr.reshape(-1, 3), axis=0), 0.0, 255.0).astype(np.uint8)
    return int(mean[0]), int(mean[1]), int(mean[2])


def map_uv_rect(uvs: np.ndarray, uv_rect: tuple) -> np.ndarray:
    """Map UVs from [0..1]^2 into a sub-rect (u0, v0, u1, v1)."""
    u0, v0, u1, v1 = uv_rect
    out = uvs.copy().astype(np.float32)
    out[:, 0] = u0 + out[:, 0] * (u1 - u0)
    out[:, 1] = v0 + out[:, 1] * (v1 - v0)
    return out


def map_capital_side_uv(verts: np.ndarray, uvs: np.ndarray, uv_rect: tuple,
                        flip_v: bool = False, wrap_per_side: bool = False) -> np.ndarray:
    """
    Remap capital UVs with height-based V so texture runs once from
    capital bottom to top. U can wrap once around or once per side.
    """
    local = uvs.copy().astype(np.float32)
    u_scale = 4.0 if wrap_per_side else 1.0
    local[:, 0] = np.mod(local[:, 0] * u_scale, 1.0)
    z = verts[:, 2].astype(np.float32)
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    dz = max(1e-6, zmax - zmin)
    local[:, 1] = (z - zmin) / dz
    if flip_v:
        local[:, 1] = 1.0 - local[:, 1]
    return map_uv_rect(local, uv_rect)


def build_texture_atlas_horizontal(main_image_path: str, extra_image_path: str,
                                   out_path: str, extra_blur_radius: float = 0.0,
                                   extra2_image_path: str = None,
                                   extra2_blur_radius: float = 0.0):
    """
    Build a texture atlas and return:
      atlas_path, main_uv_rect, extra_uv_rect, extra_blur_uv_rect, extra2_uv_rect

    Chooses horizontal or vertical packing to avoid oversized dimensions.
    """
    img_main = Image.open(main_image_path).convert("RGB")
    img_extra = Image.open(extra_image_path).convert("RGB")
    if extra_blur_radius > 0:
        img_extra_blur = img_extra.filter(ImageFilter.GaussianBlur(radius=float(extra_blur_radius)))
    else:
        img_extra_blur = img_extra
    img_extra2 = Image.open(extra2_image_path).convert("RGB") if extra2_image_path else None
    if img_extra2 is not None and extra2_blur_radius > 0:
        img_extra2 = img_extra2.filter(ImageFilter.GaussianBlur(radius=float(extra2_blur_radius)))
    mw, mh = img_main.size
    ew, eh = img_extra.size
    if img_extra2 is not None:
        e2w, e2h = img_extra2.size
    else:
        e2w, e2h = 0, 0

    # Candidate layouts:
    # Horizontal: [main | extra | extra_blur | extra2?]
    # Vertical:   [main]
    #             [extra]
    #             [extra_blur]
    #             [extra2?]
    h_aw = mw + ew + ew + e2w
    h_ah = max(mh, eh, e2h)
    v_aw = max(mw, ew, e2w)
    v_ah = mh + eh + eh + e2h

    # Conservative limit to avoid GL invalid value on texture upload.
    max_dim = 16384
    h_ok = (h_aw <= max_dim and h_ah <= max_dim)
    v_ok = (v_aw <= max_dim and v_ah <= max_dim)

    # Prefer horizontal for UV continuity, fallback to vertical when needed.
    use_horizontal = h_ok or (not v_ok and (h_aw * h_ah <= v_aw * v_ah))
    if not h_ok and v_ok:
        use_horizontal = False

    aw = h_aw if use_horizontal else v_aw
    ah = h_ah if use_horizontal else v_ah
    fill = tuple(np.asarray(img_main, dtype=np.uint8).reshape(-1, 3).mean(axis=0).astype(np.uint8).tolist())
    atlas = Image.new("RGB", (aw, ah), fill)
    if use_horizontal:
        atlas.paste(img_main, (0, 0))
        atlas.paste(img_extra, (mw, 0))
        atlas.paste(img_extra_blur, (mw + ew, 0))
        if img_extra2 is not None:
            atlas.paste(img_extra2, (mw + ew + ew, 0))
        main_rect = (0.0, 0.0, mw / aw, mh / ah)
        extra_rect = (mw / aw, 0.0, (mw + ew) / aw, eh / ah)
        extra_blur_rect = ((mw + ew) / aw, 0.0, (mw + ew + ew) / aw, eh / ah)
        if img_extra2 is not None:
            extra2_rect = ((mw + ew + ew) / aw, 0.0, 1.0, e2h / ah)
        else:
            extra2_rect = main_rect
    else:
        atlas.paste(img_main, (0, 0))
        atlas.paste(img_extra, (0, mh))
        atlas.paste(img_extra_blur, (0, mh + eh))
        if img_extra2 is not None:
            atlas.paste(img_extra2, (0, mh + eh + eh))
        main_rect = (0.0, 0.0, mw / aw, mh / ah)
        extra_rect = (0.0, mh / ah, ew / aw, (mh + eh) / ah)
        extra_blur_rect = (0.0, (mh + eh) / ah, ew / aw, (mh + eh + eh) / ah)
        if img_extra2 is not None:
            extra2_rect = (0.0, (mh + eh + eh) / ah, e2w / aw, 1.0)
        else:
            extra2_rect = main_rect

    atlas.save(out_path)
    return out_path, main_rect, extra_rect, extra_blur_rect, extra2_rect


# ──────────────────────────────────────────────────────────────────────────────
# Helical coordinate transform
# ──────────────────────────────────────────────────────────────────────────────

def helical_wrap(verts: np.ndarray, img_w_px: int, img_h_px: int,
                 helix_angle_deg: float, column_radius: float,
                 depth_scale: float, backing_depth: float) -> np.ndarray:
    """
    Maps a flat strip mesh onto a helical column surface.

    Seamless helical condition (same as helical_column_fixed.py):
      R = W_strip / (2π · sin α)
    where W_strip is the strip's SHORT dimension (img_h_px after downsampling).
    This guarantees that after one full revolution the strip has risen by
    exactly its own width — so successive turns sit flush against each other.

    Coordinate convention:
      verts[:,0]  s ∈ [0, 1]   normalised along strip LENGTH  (long axis)
      verts[:,1]  t ∈ [0, 1]   normalised across strip WIDTH  (short axis)
      verts[:,2]  z_local      radial depth displacement in world units

    Mapping (all in world units, W_strip = img_h_px * pixel_size):
      s_world = s · L_world          (L_world = img_w_px * pixel_size)
      t_world = t · W_strip_world

      x_unrolled = s_world · cos α − t_world · sin α
      Z_col      = s_world · sin α + t_world · cos α
      φ          = x_unrolled / R
      R_point    = R + z_local
      X_col      = R_point · cos φ
      Y_col      = R_point · sin φ

    pixel_size is chosen so that W_strip_world = 2π·R·sin α  (seamless condition).
    """
    α = math.radians(helix_angle_deg)
    cos_a, sin_a = math.cos(α), math.sin(α)

    # Seamless condition: W_strip_world = 2π·R·sin α
    # → pixel_size = 2π·R·sin α / img_h_px
    # → L_world    = img_w_px · pixel_size = 2π·R·sin α · (img_w_px / img_h_px)
    W_strip = 2.0 * math.pi * column_radius * sin_a          # world-unit strip width
    L_world = W_strip * (img_w_px / img_h_px)                # world-unit strip length

    s = verts[:, 0] * L_world    # [0, L_world]
    t = verts[:, 1] * W_strip    # [0, W_strip]
    z_local = verts[:, 2]

    x_unrolled = s * cos_a - t * sin_a
    Z_col      = s * sin_a + t * cos_a

    phi  = x_unrolled / column_radius
    R_pt = column_radius + z_local

    X_col = R_pt * np.cos(phi)
    Y_col = R_pt * np.sin(phi)

    return np.stack([X_col, Y_col, Z_col], axis=1).astype(np.float32)


def helical_wrap_normals(flat_normals: np.ndarray, flat_verts: np.ndarray,
                         img_w_px: int, img_h_px: int,
                         helix_angle_deg: float, column_radius: float) -> np.ndarray:
    """
    Rotates flat-space normals into column space via the Jacobian of helical_wrap.
    """
    α = math.radians(helix_angle_deg)
    cos_a, sin_a = math.cos(α), math.sin(α)

    W_strip = 2.0 * math.pi * column_radius * sin_a
    L_world = W_strip * (img_w_px / img_h_px)

    s = flat_verts[:, 0] * L_world
    t = flat_verts[:, 1] * W_strip

    x_unrolled = s * cos_a - t * sin_a
    phi = x_unrolled / column_radius

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    nx_f = flat_normals[:, 0]   # along strip length
    ny_f = flat_normals[:, 1]   # across strip width
    nz_f = flat_normals[:, 2]   # local normal (radial after wrap)

    # Tangent vectors of the helical map:
    #   ∂pos/∂s = (cos_a · (-sin_phi), cos_a · cos_phi, sin_a)   (normalised later)
    #   ∂pos/∂t = (-sin_a · (-sin_phi), -sin_a · cos_phi, cos_a)
    #   radial   = (cos_phi, sin_phi, 0)

    # ds direction (along strip length → azimuthal + vertical)
    ds_x = cos_a * (-sin_phi)
    ds_y = cos_a *   cos_phi
    ds_z = np.full_like(phi, sin_a)

    # dt direction (across strip width → helical)
    dt_x = -sin_a * (-sin_phi)
    dt_y = -sin_a *   cos_phi
    dt_z = np.full_like(phi, cos_a)

    # radial direction
    rad_x = cos_phi
    rad_y = sin_phi
    rad_z = np.zeros_like(phi)

    # Rotate normal
    Nx = nx_f * ds_x + ny_f * dt_x + nz_f * rad_x
    Ny = nx_f * ds_y + ny_f * dt_y + nz_f * rad_y
    Nz = nx_f * ds_z + ny_f * dt_z + nz_f * rad_z

    out = np.stack([Nx, Ny, Nz], axis=1).astype(np.float32)
    nlen = np.linalg.norm(out, axis=1, keepdims=True)
    nlen = np.where(nlen == 0, 1.0, nlen)
    return out / nlen


# ──────────────────────────────────────────────────────────────────────────────
# Mesh construction
# ──────────────────────────────────────────────────────────────────────────────

def build_mesh(colour: np.ndarray, depth: np.ndarray,
               depth_scale: float, backing_depth: float = 0.0,
               wrap_column: bool = False,
               helix_angle_deg: float = 15.0,
               column_radius: float = None,
               add_lip: bool = False,
               lip_height: float = None,
               lip_depth: float = 0.0,
               end_fade_enabled: bool = False,
               end_fade_fraction: float = 0.0,
               end_fade_power: float = 1.0,
               mirror_edge_enabled: bool = False,
               mirror_edge_fraction: float = 0.10):
    """
    Builds a solid displaced-plane mesh (plaque / bas-relief).

    Flat mode  (wrap_column=False):
      The mesh spans [-aspect, +aspect] in X and [-1, +1] in Y.
      Z is displaced by depth_scale * depth_value on the front face.

    Column mode  (wrap_column=True):
      The flat strip is built in pixel-space (x ∈ [0, W], y ∈ [0, H]),
      then every vertex is passed through helical_wrap() which bends the
      strip around a vertical cylinder at the given helix angle.
      The depth displacement (z_local) becomes radial displacement on the
      column surface — raised areas stick outward, exactly like carved relief.
      The back face sits at R - backing_depth (inner cylinder surface).

    When backing_depth > 0 the mesh is closed into a solid by adding:
      • a back face (flat plate in flat mode; inner cylinder strip in column mode)
      • four side walls (helical edges in column mode)

    Returns
    -------
    vertices  : (N, 3) float32
    normals   : (N, 3) float32
    uvs       : (N, 2) float32
    colours   : (N, 3) uint8
    indices   : (M, 3) int32
    col_height : float   (column height in world units, only meaningful in column mode)
    lip_tri_mask : (M,) bool  triangles to render as stone lip in preview
    """
    H_img, W = depth.shape
    depth_work = depth.astype(np.float32)
    colour_work = colour
    lip_rows = 0
    lip_t_span = 0.0
    end_fade_profile = np.ones(W, dtype=np.float32)

    # Mirror extension strips at helix start/end (adds geometry outside s=[0,1]).
    n_mirror = 0
    if wrap_column and mirror_edge_enabled and mirror_edge_fraction > 0:
        n_mirror = int(round(W * float(mirror_edge_fraction)))
        n_mirror = max(1, min(n_mirror, max(1, (W // 2) - 1)))
        if n_mirror > 0:
            print(f"[mirror] Edge mirror strips enabled: {n_mirror} cols each side")

    use_end_fade = wrap_column and end_fade_enabled and end_fade_fraction > 0
    if wrap_column and mirror_edge_enabled and mirror_edge_fraction > 0:
        use_end_fade = False
        if end_fade_enabled and end_fade_fraction > 0:
            print("[fade] Skipping end fade because mirror-edge strips are enabled")
    if use_end_fade:
        n_fade = int(round(W * float(end_fade_fraction)))
        n_fade = max(0, min(n_fade, W // 2))
        if n_fade > 0:
            ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
            end_fade_profile[:n_fade] = ramp
            end_fade_profile[-n_fade:] = np.minimum(end_fade_profile[-n_fade:], ramp[::-1])
            if end_fade_power != 1.0:
                end_fade_profile = np.power(end_fade_profile, float(end_fade_power))
            depth_work = depth_work * end_fade_profile[np.newaxis, :]
            print(f"[fade] End taper enabled: {n_fade} cols each side (fraction={end_fade_fraction:.4f}, power={end_fade_power:.3f})")

    if wrap_column and add_lip and lip_depth > 0:
        alpha = math.radians(helix_angle_deg)
        R = 1.0 if column_radius is None else float(column_radius)
        if abs(math.cos(alpha)) < 1e-6:
            raise ValueError("[lip] helix angle too close to 90 deg")

        # Base column height (without lip) drives the 1/20 default.
        w_strip_base = 2.0 * math.pi * R * math.sin(alpha)
        l_world_base = w_strip_base * (W / H_img)
        base_col_height = l_world_base * math.sin(alpha) + w_strip_base * math.cos(alpha)
        lip_h_world = (base_col_height / 20.0) if lip_height is None else float(lip_height)
        if lip_h_world <= 0:
            raise ValueError(f"[lip] Lip height must be > 0 (got {lip_h_world})")

        # Convert target vertical lip height (column Z) into helical strip t-span.
        lip_t_world = lip_h_world / math.cos(alpha)
        lip_t_span = lip_t_world / w_strip_base
        lip_rows = max(1, int(round(H_img * lip_t_span)))

        lip_z = np.full((lip_rows, W), float(lip_depth), dtype=np.float32)
        lip_z *= end_fade_profile[np.newaxis, :]
        depth_work = np.concatenate([depth_work * depth_scale, lip_z], axis=0)
        lip_col = np.repeat(colour[-1:, :, :], lip_rows, axis=0)
        colour_work = np.concatenate([colour, lip_col], axis=0)

        print(f"[lip] Integrated helical divider: h={lip_h_world:.4f} depth={lip_depth:.4f} rows={lip_rows}")
    else:
        depth_work = depth_work * depth_scale

    H = depth_work.shape[0]

    if wrap_column:
        # Normalised [0,1] coordinates.
        # s=0 → left edge of image (column base, φ=0)
        # s=1 → right edge (one full revolution + height gain)
        # t=0 → bottom of strip (row H-1 in image = bottom edge)
        # t=1 → top of strip   (row 0 in image = top edge)
        # Image row 0 is the top, so we flip y so t=0 is the bottom of the strip.
        xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
        ys = np.linspace(1.0, 0.0, H, dtype=np.float32)  # row 0 → t=1 (top), row H-1 → t=0 (bottom)
        if lip_rows > 0:
            ys_img = ys[:H_img]
            ys_lip = -((np.arange(1, lip_rows + 1, dtype=np.float32) / lip_rows) * lip_t_span)
            ys = np.concatenate([ys_img, ys_lip], axis=0)
    else:
        aspect = W / H
        xs = np.linspace(-aspect, aspect, W, dtype=np.float32)
        ys = np.linspace( 1.0,   -1.0,   H, dtype=np.float32)

    xg, yg = np.meshgrid(xs, ys)
    zg = depth_work.astype(np.float32)

    # ── front face ────────────────────────────────────────────────────────
    front_verts = np.stack([xg, yg, zg], axis=-1).reshape(-1, 3)
    u_grid = np.linspace(0, 1, W, dtype=np.float32)[np.newaxis, :].repeat(H, 0)
    if wrap_column and lip_rows > 0:
        v_img = np.linspace(0, 1, H_img, dtype=np.float32)
        strip = max(1e-4, min(1.0, float(LIP_TEXTURE_STRIP_FRACTION)))
        v_lip = np.linspace(1.0 - strip, 1.0, lip_rows, dtype=np.float32)
        v_all = np.concatenate([v_img, v_lip], axis=0)
    else:
        v_all = np.linspace(0, 1, H, dtype=np.float32)
    v_grid = v_all[:, np.newaxis].repeat(W, 1)
    front_uvs   = np.stack([u_grid, v_grid], axis=-1).reshape(-1, 2)
    front_cols  = colour_work.reshape(-1, 3)

    grid = np.arange(H * W, dtype=np.int32).reshape(H, W)
    tl = grid[:-1, :-1].ravel()
    tr = grid[:-1,  1:].ravel()
    bl = grid[ 1:, :-1].ravel()
    br = grid[ 1:,  1:].ravel()
    front_tris = np.concatenate([
        np.stack([tl, bl, tr], axis=1),
        np.stack([tr, bl, br], axis=1),
    ], axis=0)

    front_normals = compute_normals(front_verts, front_tris, H * W)
    front_lip_mask = np.zeros(len(front_tris), dtype=bool)
    if wrap_column and lip_rows > 0:
        # Mark front triangles that belong to lip rows (including transition row).
        tri_rows = (front_tris // W)
        front_lip_mask = np.max(tri_rows, axis=1) >= H_img

    all_verts   = [front_verts]
    all_normals = [front_normals]
    all_uvs     = [front_uvs]
    all_cols    = [front_cols]
    all_tris    = [front_tris]
    offset      = H * W          # running vertex offset

    if backing_depth > 0.0:
        z_back = -backing_depth

        # ── back face ─────────────────────────────────────────────────────
        back_verts   = np.stack([xg, yg,
                                 np.full_like(zg, z_back)], axis=-1).reshape(-1, 3)
        back_normals_flat = np.zeros_like(back_verts)
        back_normals_flat[:, 2] = -1.0
        back_uvs  = front_uvs.copy()
        back_cols = front_cols.copy()

        # Flip winding so normals face outward (-Z / inward radially)
        back_tris = np.concatenate([
            np.stack([tl, tr, bl], axis=1),
            np.stack([tr, br, bl], axis=1),
        ], axis=0) + offset

        all_verts.append(back_verts)
        all_normals.append(back_normals_flat)
        all_uvs.append(back_uvs)
        all_cols.append(back_cols)
        all_tris.append(back_tris)
        offset += H * W

        # ── side walls ────────────────────────────────────────────────────
        def make_wall(front_edge, back_edge, outward_normal_flat):
            K = len(front_edge)
            fl  = front_edge[:-1]
            fr  = front_edge[1:]
            bl_ = back_edge[:-1]
            br_ = back_edge[1:]

            wv = np.concatenate([fl, fr, bl_, br_], axis=0)
            n_quads = K - 1
            base = np.arange(n_quads, dtype=np.int32)
            i_fl = base
            i_fr = base + n_quads
            i_bl = base + 2 * n_quads
            i_br = base + 3 * n_quads
            wt = np.concatenate([
                np.stack([i_fl, i_bl, i_fr], axis=1),
                np.stack([i_fr, i_bl, i_br], axis=1),
            ], axis=0)

            wn = np.tile(outward_normal_flat.astype(np.float32), (len(wv), 1))
            u_e = np.linspace(0, 1, K - 1, dtype=np.float32)
            wu = np.concatenate([
                np.stack([u_e, np.zeros(n_quads, dtype=np.float32)], axis=1),
                np.stack([u_e, np.zeros(n_quads, dtype=np.float32)], axis=1),
                np.stack([u_e, np.ones (n_quads, dtype=np.float32)], axis=1),
                np.stack([u_e, np.ones (n_quads, dtype=np.float32)], axis=1),
            ], axis=0)
            wc = np.tile(np.array([128, 128, 128], dtype=np.uint8), (len(wv), 1))
            return wv, wn, wu, wc, wt

        fv = front_verts.reshape(H, W, 3)

        # Top edge (row 0, L→R), outward = +Y in flat space
        top_f = fv[0, :, :]
        top_b = np.stack([top_f[:, 0], top_f[:, 1],
                          np.full(W, z_back, dtype=np.float32)], axis=1)
        wv, wn, wu, wc, wt = make_wall(top_f, top_b, np.array([0, 1, 0]))
        all_verts.append(wv); all_normals.append(wn)
        all_uvs.append(wu);   all_cols.append(wc)
        all_tris.append(wt + offset); offset += len(wv)

        # Bottom edge (row -1, reversed R→L), outward = -Y
        bot_f = fv[-1, :, :]
        bot_b = np.stack([bot_f[:, 0], bot_f[:, 1],
                          np.full(W, z_back, dtype=np.float32)], axis=1)
        wv, wn, wu, wc, wt = make_wall(bot_f[::-1], bot_b[::-1], np.array([0, -1, 0]))
        all_verts.append(wv); all_normals.append(wn)
        all_uvs.append(wu);   all_cols.append(wc)
        all_tris.append(wt + offset); offset += len(wv)

        # Right edge (col -1, T→B), outward = +X
        rgt_f = fv[:, -1, :]
        rgt_b = np.stack([rgt_f[:, 0], rgt_f[:, 1],
                          np.full(H, z_back, dtype=np.float32)], axis=1)
        wv, wn, wu, wc, wt = make_wall(rgt_f, rgt_b, np.array([1, 0, 0]))
        all_verts.append(wv); all_normals.append(wn)
        all_uvs.append(wu);   all_cols.append(wc)
        all_tris.append(wt + offset); offset += len(wv)

        # Left edge (col 0, reversed B→T), outward = -X
        lft_f = fv[:, 0, :]
        lft_b = np.stack([lft_f[:, 0], lft_f[:, 1],
                          np.full(H, z_back, dtype=np.float32)], axis=1)
        wv, wn, wu, wc, wt = make_wall(lft_f[::-1], lft_b[::-1], np.array([-1, 0, 0]))
        all_verts.append(wv); all_normals.append(wn)
        all_uvs.append(wu);   all_cols.append(wc)
        all_tris.append(wt + offset); offset += len(wv)

    # Additional mirrored helical strips before s=0 and after s=1 to fill
    # the natural start/end sparse zones without changing base/endcap alignment.
    if wrap_column and n_mirror > 0 and W > 1:
        x_step = xs[1] - xs[0]
        v_vals = v_all

        def make_mirror_strip(x_vals, depth_cols, colour_cols, u_vals):
            xg_e, yg_e = np.meshgrid(x_vals, ys)
            zg_e = depth_cols.astype(np.float32)
            verts_e = np.stack([xg_e, yg_e, zg_e], axis=-1).reshape(-1, 3)
            u_grid_e = u_vals[np.newaxis, :].repeat(H, 0)
            v_grid_e = v_vals[:, np.newaxis].repeat(len(x_vals), 1)
            uvs_e = np.stack([u_grid_e, v_grid_e], axis=-1).reshape(-1, 2).astype(np.float32)
            cols_e = colour_cols.reshape(-1, 3).astype(np.uint8)

            grid_e = np.arange(H * len(x_vals), dtype=np.int32).reshape(H, len(x_vals))
            tl_e = grid_e[:-1, :-1].ravel()
            tr_e = grid_e[:-1,  1:].ravel()
            bl_e = grid_e[ 1:, :-1].ravel()
            br_e = grid_e[ 1:,  1:].ravel()
            tris_e = np.concatenate([
                np.stack([tl_e, bl_e, tr_e], axis=1),
                np.stack([tr_e, bl_e, br_e], axis=1),
            ], axis=0).astype(np.int32)
            normals_e = compute_normals(verts_e, tris_e, len(verts_e))
            return verts_e, normals_e, uvs_e, cols_e, tris_e

        # Start strip (before s=0): mirror first columns.
        x_left = -np.arange(n_mirror, 0, -1, dtype=np.float32) * x_step
        d_left = depth_work[:, :n_mirror][:, ::-1]
        c_left = colour_work[:, :n_mirror, :][:, ::-1, :]
        u_left_idx = np.arange(n_mirror - 1, -1, -1, dtype=np.float32)
        u_left = u_left_idx / max(W - 1, 1)
        v_e, n_e, uv_e, c_e, t_e = make_mirror_strip(x_left, d_left, c_left, u_left)
        all_verts.append(v_e); all_normals.append(n_e); all_uvs.append(uv_e); all_cols.append(c_e)
        all_tris.append(t_e + offset); offset += len(v_e)

        # End strip (after s=1): mirror last columns.
        x_right = 1.0 + np.arange(1, n_mirror + 1, dtype=np.float32) * x_step
        d_right = depth_work[:, W - n_mirror:W][:, ::-1]
        c_right = colour_work[:, W - n_mirror:W, :][:, ::-1, :]
        u_right_idx = np.arange(W - 1, W - 1 - n_mirror, -1, dtype=np.float32)
        u_right = u_right_idx / max(W - 1, 1)
        v_e, n_e, uv_e, c_e, t_e = make_mirror_strip(x_right, d_right, c_right, u_right)
        all_verts.append(v_e); all_normals.append(n_e); all_uvs.append(uv_e); all_cols.append(c_e)
        all_tris.append(t_e + offset); offset += len(v_e)

    # ── combine flat mesh ─────────────────────────────────────────────────
    flat_verts   = np.concatenate(all_verts,   axis=0).astype(np.float32)
    flat_normals = np.concatenate(all_normals, axis=0).astype(np.float32)
    uvs          = np.concatenate(all_uvs,     axis=0).astype(np.float32)
    cols         = np.concatenate(all_cols,    axis=0).astype(np.uint8)
    indices      = np.concatenate(all_tris,    axis=0).astype(np.int32)
    lip_tri_mask = np.zeros(len(indices), dtype=bool)
    lip_tri_mask[:len(front_tris)] = front_lip_mask

    # ── helical wrap ──────────────────────────────────────────────────────
    col_height = 0.0
    if wrap_column:
        α = math.radians(helix_angle_deg)
        R = 1.0 if column_radius is None else float(column_radius)

        # Seamless condition: W_strip = 2π·R·sin α
        # Column height = L_world·sin α + W_strip·cos α
        W_strip = 2.0 * math.pi * R * math.sin(α)
        L_world = W_strip * (W / H)          # W=mesh cols (length), H=mesh rows (width)
        col_height = L_world * math.sin(α) + W_strip * math.cos(α)
        n_turns = L_world / (2.0 * math.pi * R)

        print(f"[column] Helix angle  : {helix_angle_deg}°")
        print(f"[column] Radius       : {R:.4f}  (diameter {2*R:.4f})")
        print(f"[column] Strip width  : {W_strip:.4f}  length {L_world:.4f}")
        print(f"[column] Turns        : {n_turns:.2f}")
        print(f"[column] Column height: {col_height:.4f}")
        print(f"[column] D/H ratio    : {2*R/col_height:.4f}")

        verts   = helical_wrap(flat_verts, W, H, helix_angle_deg, R,
                               depth_scale, backing_depth)
        normals = helical_wrap_normals(flat_normals, flat_verts, W, H,
                                       helix_angle_deg, R)

        # Hard-trim any helix triangles that protrude above the endcap plane.
        if CLIP_HELIX_AT_ENDCAP and len(indices) > 0:
            z_tri_max = np.max(verts[indices][:, :, 2], axis=1)
            keep = z_tri_max <= (col_height + float(HELIX_TOP_CLIP_EPS))
            if not np.all(keep):
                removed = int(np.count_nonzero(~keep))
                indices = indices[keep]
                lip_tri_mask = lip_tri_mask[keep]
                print(f"[clip] Removed {removed} triangles above endcap plane")
    else:
        verts   = flat_verts
        normals = flat_normals

    print(f"[mesh] {'Column' if wrap_column else 'Flat'} solid: "
          f"{len(verts):,} vertices  |  {len(indices):,} triangles"
          + (f"  (backing {backing_depth:.3f})" if backing_depth > 0 else ""))

    return verts, normals, uvs, cols, indices, col_height, lip_tri_mask


def compute_normals(verts: np.ndarray, indices: np.ndarray, n_verts: int):
    """Accumulate face normals into per-vertex normals."""
    v0 = verts[indices[:, 0]]
    v1 = verts[indices[:, 1]]
    v2 = verts[indices[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros((n_verts, 3), dtype=np.float32)
    for i in range(3):
        np.add.at(normals, indices[:, i], fn)

    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen = np.where(nlen == 0, 1.0, nlen)
    return (normals / nlen).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Column parts  (inner cylinder, base, endcap)
# Each function returns (verts, normals, uvs, cols, indices) ready to be
# concatenated with any other mesh via combine_meshes().
# ──────────────────────────────────────────────────────────────────────────────

def combine_meshes(*parts):
    """
    Concatenate any number of (verts, normals, uvs, cols, indices) tuples
    into a single mesh, adjusting index offsets automatically.
    """
    all_v, all_n, all_u, all_c, all_i = [], [], [], [], []
    offset = 0
    for v, n, u, c, idx in parts:
        all_v.append(v)
        all_n.append(n)
        all_u.append(u)
        all_c.append(c)
        all_i.append(idx + offset)
        offset += len(v)
    return (
        np.concatenate(all_v,  axis=0).astype(np.float32),
        np.concatenate(all_n,  axis=0).astype(np.float32),
        np.concatenate(all_u,  axis=0).astype(np.float32),
        np.concatenate(all_c,  axis=0).astype(np.uint8),
        np.concatenate(all_i,  axis=0).astype(np.int32),
    )


def load_glb_mesh(path: str, colour: tuple = (180, 160, 120)):
    """
    Load GLB mesh(es) and first available texture.
    Returns:
      (verts, normals, uvs, cols, indices, tex_rgb_or_none)
    """
    try:
        import trimesh
    except ImportError:
        raise RuntimeError("[parts] trimesh is required for GLB statue loading (pip install trimesh[easy])")

    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        meshes = []
        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                meshes.append(g)

    if not meshes:
        raise ValueError(f"[parts] GLB has no mesh geometry: {path}")

    all_parts = []
    tex_rgb = None
    col = np.array(colour, dtype=np.uint8)

    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float32)
        i = np.asarray(m.faces, dtype=np.int32)
        if len(v) == 0 or len(i) == 0:
            continue

        if m.vertex_normals is not None and len(m.vertex_normals) == len(v):
            n = np.asarray(m.vertex_normals, dtype=np.float32)
        else:
            n = compute_normals(v, i, len(v))

        uv = None
        visual = getattr(m, "visual", None)
        if visual is not None and hasattr(visual, "uv") and visual.uv is not None and len(visual.uv) == len(v):
            uv = np.asarray(visual.uv, dtype=np.float32)
        if uv is None:
            x = v[:, 0]
            z = v[:, 2]
            dx = max(1e-6, float(np.max(x) - np.min(x)))
            dz = max(1e-6, float(np.max(z) - np.min(z)))
            uv = np.stack([(x - np.min(x)) / dx, (z - np.min(z)) / dz], axis=1).astype(np.float32)

        if tex_rgb is None and visual is not None and hasattr(visual, "material"):
            mat = visual.material
            img = getattr(mat, "baseColorTexture", None)
            if img is not None:
                tex_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            else:
                img2 = getattr(mat, "image", None)
                if img2 is not None:
                    tex_rgb = np.asarray(img2.convert("RGB"), dtype=np.uint8)

        c = np.tile(col, (len(v), 1))
        all_parts.append((v, n, uv, c, i))

    if not all_parts:
        raise ValueError(f"[parts] GLB meshes were empty: {path}")
    v, n, uv, c, i = combine_meshes(*all_parts)
    return v, n, uv, c, i, tex_rgb


def build_inner_cylinder(radius: float, height: float,
                         segments: int = 64,
                         colour: tuple = (180, 160, 120)) -> tuple:
    """
    Solid cylinder from Z=0 to Z=height, radius=radius.
    Consists of: outer wall, bottom disc, top disc.
    Normals point outward on the wall and ±Z on the caps.
    colour : RGB tuple used for all vertices (stone/marble colour by default).
    """
    print(f"[parts] Inner cylinder  r={radius:.4f}  h={height:.4f}  seg={segments}")
    col = np.array(colour, dtype=np.uint8)

    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False, dtype=np.float32)
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    all_v, all_n, all_u, all_c, all_i = [], [], [], [], []

    # ── outer wall ────────────────────────────────────────────────────────
    # 2 rings of vertices (bottom and top), each with outward normals
    bot = np.stack([radius * cos_a, radius * sin_a, np.zeros(segments)], axis=1)
    top = np.stack([radius * cos_a, radius * sin_a, np.full(segments, height)], axis=1)
    wall_v = np.concatenate([bot, top], axis=0).astype(np.float32)   # (2S, 3)
    wall_n = np.concatenate([
        np.stack([cos_a, sin_a, np.zeros(segments)], axis=1),
        np.stack([cos_a, sin_a, np.zeros(segments)], axis=1),
    ], axis=0).astype(np.float32)
    wall_u = np.concatenate([
        np.stack([angles / (2 * math.pi), np.zeros(segments)], axis=1),
        np.stack([angles / (2 * math.pi), np.ones(segments)],  axis=1),
    ], axis=0).astype(np.float32)
    wall_c = np.tile(col, (len(wall_v), 1))

    idx = np.arange(segments, dtype=np.int32)
    next_idx = (idx + 1) % segments
    # bottom ring = [0..S-1], top ring = [S..2S-1]
    wall_i = np.concatenate([
        np.stack([idx, next_idx, idx + segments], axis=1),
        np.stack([next_idx, next_idx + segments, idx + segments], axis=1),
    ], axis=0)

    all_v.append(wall_v); all_n.append(wall_n)
    all_u.append(wall_u); all_c.append(wall_c)
    all_i.append(wall_i)

    # ── bottom cap (normal = -Z) ───────────────────────────────────────────
    rim_b = np.stack([radius * cos_a, radius * sin_a, np.zeros(segments)], axis=1)
    ctr_b = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cap_b_v = np.concatenate([rim_b, ctr_b], axis=0).astype(np.float32)
    cap_b_n = np.tile([0.0, 0.0, -1.0], (len(cap_b_v), 1)).astype(np.float32)
    cap_b_u = np.zeros((len(cap_b_v), 2), dtype=np.float32)
    cap_b_c = np.tile(col, (len(cap_b_v), 1))
    ctr_i   = segments   # index of centre vertex
    cap_b_i = np.stack([idx, ctr_i * np.ones(segments, dtype=np.int32), next_idx], axis=1)

    all_v.append(cap_b_v); all_n.append(cap_b_n)
    all_u.append(cap_b_u); all_c.append(cap_b_c)
    all_i.append(cap_b_i)

    # ── top cap (normal = +Z) ──────────────────────────────────────────────
    rim_t = np.stack([radius * cos_a, radius * sin_a, np.full(segments, height)], axis=1)
    ctr_t = np.array([[0.0, 0.0, height]], dtype=np.float32)
    cap_t_v = np.concatenate([rim_t, ctr_t], axis=0).astype(np.float32)
    cap_t_n = np.tile([0.0, 0.0, 1.0], (len(cap_t_v), 1)).astype(np.float32)
    cap_t_u = np.zeros((len(cap_t_v), 2), dtype=np.float32)
    cap_t_c = np.tile(col, (len(cap_t_v), 1))
    cap_t_i = np.stack([idx, next_idx, ctr_i * np.ones(segments, dtype=np.int32)], axis=1)

    all_v.append(cap_t_v); all_n.append(cap_t_n)
    all_u.append(cap_t_u); all_c.append(cap_t_c)
    all_i.append(cap_t_i)

    return combine_meshes(*zip(all_v, all_n, all_u, all_c, all_i))


def _build_box_plinth(half_extent: float, z0: float, z1: float,
                      colour: tuple = (160, 140, 100),
                      uv_rect: tuple = (0.0, 0.0, 1.0, 1.0)) -> tuple:
    """
    Rectangular box centred on the column axis, flat-shaded per face.
    Spans Z in [z0, z1] with half-width = half_extent in X and Y.
    """
    r = float(half_extent)
    if z1 <= z0:
        raise ValueError(f"[parts] Invalid box Z range: z0={z0}, z1={z1}")
    col = np.array(colour, dtype=np.uint8)

    corners = np.array([
        [-r, -r, z0], [ r, -r, z0], [ r,  r, z0], [-r,  r, z0],
        [-r, -r, z1], [ r, -r, z1], [ r,  r, z1], [-r,  r, z1],
    ], dtype=np.float32)

    faces = [
        ([4,5,6,7],  [ 0,  0,  1]),
        ([0,3,2,1],  [ 0,  0, -1]),
        ([0,1,5,4],  [ 0, -1,  0]),
        ([2,3,7,6],  [ 0,  1,  0]),
        ([1,2,6,5],  [ 1,  0,  0]),
        ([3,0,4,7],  [-1,  0,  0]),
    ]

    all_v, all_n, all_u, all_c, all_i = [], [], [], [], []
    for vi, nrm in faces:
        fv = corners[vi]
        fn = np.tile(nrm, (4, 1)).astype(np.float32)
        fu = map_uv_rect(np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32), uv_rect)
        fc = np.tile(col, (4, 1))
        fi = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        all_v.append(fv); all_n.append(fn)
        all_u.append(fu); all_c.append(fc); all_i.append(fi)

    return combine_meshes(*zip(all_v, all_n, all_u, all_c, all_i))


def _sample_bilinear_rgb(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample RGB image with UV in [0,1]. Returns uint8 RGB."""
    h, w, _ = img.shape
    uu = np.clip(u, 0.0, 1.0) * (w - 1)
    vv = np.clip(v, 0.0, 1.0) * (h - 1)
    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (uu - x0).astype(np.float32)
    fy = (vv - y0).astype(np.float32)

    c00 = img[y0, x0].astype(np.float32)
    c10 = img[y0, x1].astype(np.float32)
    c01 = img[y1, x0].astype(np.float32)
    c11 = img[y1, x1].astype(np.float32)
    c0 = c00 * (1.0 - fx[:, None]) + c10 * fx[:, None]
    c1 = c01 * (1.0 - fx[:, None]) + c11 * fx[:, None]
    c = c0 * (1.0 - fy[:, None]) + c1 * fy[:, None]
    return np.clip(np.round(c), 0.0, 255.0).astype(np.uint8)


def _sample_bilinear_gray(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample grayscale image with UV in [0,1]. Returns float32 in [0,1]."""
    h, w = img.shape
    uu = np.clip(u, 0.0, 1.0) * (w - 1)
    vv = np.clip(v, 0.0, 1.0) * (h - 1)
    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (uu - x0).astype(np.float32)
    fy = (vv - y0).astype(np.float32)

    c00 = img[y0, x0].astype(np.float32)
    c10 = img[y0, x1].astype(np.float32)
    c01 = img[y1, x0].astype(np.float32)
    c11 = img[y1, x1].astype(np.float32)
    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return np.clip(c0 * (1.0 - fy) + c1 * fy, 0.0, 1.0)


def _build_textured_top_plinth(half_extent: float, z0: float, z1: float,
                               texture_rgb: np.ndarray, depth_gray: np.ndarray,
                               depth_scale: float,
                               u_segments: int, v_segments: int,
                               base_colour: tuple,
                               uv_rect: tuple,
                               flip_v: bool = False,
                               depth_invert: bool = False,
                               recess: float = 0.0,
                               corner_blend_fraction: float = 0.08) -> tuple:
    """
    Build top plinth with side-band relief from texture/depth and flat top/bottom caps.
    """
    r = float(half_extent)
    h = float(z1 - z0)
    if h <= 0:
        raise ValueError(f"[parts] Invalid textured plinth height: {h}")
    if u_segments < 16 or v_segments < 2:
        raise ValueError("[parts] Plinth side segments too low")

    # Build four side strips as one perimeter unwrap in U, but map each face
    # to a full [0..1] texture span so each side gets the complete plinth motif.
    perim = 8.0 * r
    side_u = max(16, int(u_segments))
    side_v = max(2, int(v_segments))
    u_line = np.linspace(0.0, 1.0, side_u + 1, dtype=np.float32)
    v_line = np.linspace(0.0, 1.0, side_v + 1, dtype=np.float32)
    uu, vv = np.meshgrid(u_line, v_line, indexing="xy")
    uu_f = uu.ravel()
    vv_f = vv.ravel()

    face_u = np.empty_like(uu_f, dtype=np.float32)
    s = uu_f * perim
    z = z0 + vv_f * h

    x = np.empty_like(s, dtype=np.float32)
    y = np.empty_like(s, dtype=np.float32)
    nx = np.empty_like(s, dtype=np.float32)
    ny = np.empty_like(s, dtype=np.float32)

    m0 = s < (2.0 * r)                           # front (-Y)
    m1 = (s >= 2.0 * r) & (s < 4.0 * r)          # right (+X)
    m2 = (s >= 4.0 * r) & (s < 6.0 * r)          # back (+Y)
    m3 = ~ (m0 | m1 | m2)                        # left (-X)

    t0 = np.zeros_like(s); t0[m0] = s[m0] / (2.0 * r)
    x[m0] = -r + 2.0 * r * t0[m0]
    face_u[m0] = t0[m0]
    nx[m0], ny[m0] = 0.0, -1.0

    t1 = np.zeros_like(s); t1[m1] = (s[m1] - 2.0 * r) / (2.0 * r)
    face_u[m1] = t1[m1]
    y[m1] = -r + 2.0 * r * t1[m1]
    nx[m1], ny[m1] = 1.0, 0.0

    t2 = np.zeros_like(s); t2[m2] = (s[m2] - 4.0 * r) / (2.0 * r)
    face_u[m2] = t2[m2]
    x[m2] = r - 2.0 * r * t2[m2]
    nx[m2], ny[m2] = 0.0, 1.0

    t3 = np.zeros_like(s); t3[m3] = (s[m3] - 6.0 * r) / (2.0 * r)
    face_u[m3] = t3[m3]
    y[m3] = r - 2.0 * r * t3[m3]
    nx[m3], ny[m3] = -1.0, 0.0

    # Keep color/depth orientation configurable per source image.
    sample_v = (1.0 - vv_f) if flip_v else vv_f
    depth = _sample_bilinear_gray(depth_gray, face_u, sample_v)
    if depth_invert:
        depth = 1.0 - depth
    disp = depth * float(depth_scale)
    blend_w = max(1e-4, float(corner_blend_fraction))
    edge_blend_u = np.clip(np.minimum(face_u, 1.0 - face_u) / blend_w, 0.0, 1.0)
    edge_blend_v = np.clip(np.minimum(vv_f, 1.0 - vv_f) / blend_w, 0.0, 1.0)
    disp = disp * edge_blend_u * edge_blend_v
    r_side = max(1e-5, r - float(recess))
    y[m0] = -r_side - disp[m0]
    x[m1] = r_side + disp[m1]
    y[m2] = r_side + disp[m2]
    x[m3] = -r_side - disp[m3]

    side_vtx = np.stack([x, y, z], axis=1).astype(np.float32)
    side_nrm = np.stack([nx, ny, np.zeros_like(nx)], axis=1).astype(np.float32)
    side_uv = map_uv_rect(np.stack([face_u, sample_v], axis=1).astype(np.float32), uv_rect)
    side_col = _sample_bilinear_rgb(texture_rgb, face_u, sample_v)

    grid = np.arange((side_v + 1) * (side_u + 1), dtype=np.int32).reshape(side_v + 1, side_u + 1)
    a = grid[:-1, :-1].ravel()
    b = grid[:-1, 1:].ravel()
    c = grid[1:, :-1].ravel()
    d = grid[1:, 1:].ravel()
    side_idx = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([c, b, d], axis=1)], axis=0
    ).astype(np.int32)

    # Flat top and bottom caps.
    cap_col = np.array(base_colour, dtype=np.uint8)
    r_cap = r_side
    cap_uv = map_uv_rect(np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32), uv_rect)
    top_v = np.array([[-r_cap, -r_cap, z1], [r_cap, -r_cap, z1], [r_cap, r_cap, z1], [-r_cap, r_cap, z1]], dtype=np.float32)
    top_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    top_c = np.tile(cap_col, (4, 1))
    top_i = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    bot_v = np.array([[-r_cap, -r_cap, z0], [-r_cap, r_cap, z0], [r_cap, r_cap, z0], [r_cap, -r_cap, z0]], dtype=np.float32)
    bot_n = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (4, 1))
    bot_c = np.tile(cap_col, (4, 1))
    bot_i = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    return combine_meshes(
        (side_vtx, side_nrm, side_uv, side_col, side_idx),
        (top_v, top_n, cap_uv, top_c, top_i),
        (bot_v, bot_n, cap_uv, bot_c, bot_i),
    )


def build_block_base(column_radius: float, col_z_bottom: float,
                     overhang: float = 0.3, height: float = 0.4,
                     colour: tuple = (160, 140, 100),
                     uv_rect: tuple = (0.0, 0.0, 1.0, 1.0)) -> tuple:
    """
    Single rectangular plinth under the column.
    Sits from Z = col_z_bottom - height to Z = col_z_bottom.
    """
    r = column_radius + overhang
    z0 = col_z_bottom - height
    z1 = col_z_bottom
    print(f"[parts] Base(block)  +/-{r:.4f}  z=[{z0:.4f}, {z1:.4f}]")
    return _build_box_plinth(r, z0, z1, colour, uv_rect=uv_rect)


def build_stepped_base(column_radius: float, col_z_bottom: float,
                       overhang: float = 0.3, total_height: float = 1.5,
                       top_plinth_height: float = 0.5, steps: int = 6,
                       step_run: float = 0.25, step_height: float = None,
                       top_steps: int = 0, top_step_run: float = 0.1,
                       top_step_height: float = 0.08,
                       top_plinth_texture_rgb: np.ndarray = None,
                       top_plinth_depth_gray: np.ndarray = None,
                       top_plinth_depth_scale: float = 0.03,
                       top_plinth_u_segments: int = 128,
                       top_plinth_v_segments: int = 16,
                       top_plinth_flip_v: bool = False,
                       top_plinth_depth_invert: bool = False,
                       top_plinth_recess: float = 0.0,
                       top_plinth_corner_blend_fraction: float = 0.08,
                       uv_rect_base: tuple = (0.0, 0.0, 1.0, 1.0),
                       uv_rect_top_plinth: tuple = (0.0, 0.0, 1.0, 1.0),
                       colour: tuple = (160, 140, 100)) -> tuple:
    """
    Monument-style stepped base.
    The top plinth touches the column at col_z_bottom, then tiers descend.
    """
    if steps < 1:
        raise ValueError(f"[parts] BASE_STEPS must be >= 1 (got {steps})")
    if top_plinth_height <= 0:
        raise ValueError(f"[parts] BASE_TOP_PLINTH_HEIGHT must be > 0 (got {top_plinth_height})")
    if step_run <= 0:
        raise ValueError(f"[parts] BASE_STEP_RUN must be > 0 (got {step_run})")
    if top_steps < 0:
        raise ValueError(f"[parts] BASE_TOP_STEPS must be >= 0 (got {top_steps})")
    if top_steps > 0 and top_step_run <= 0:
        raise ValueError(f"[parts] BASE_TOP_STEP_RUN must be > 0 (got {top_step_run})")
    if top_steps > 0 and top_step_height <= 0:
        raise ValueError(f"[parts] BASE_TOP_STEP_HEIGHT must be > 0 (got {top_step_height})")

    top_r = column_radius + overhang
    if step_height is None:
        remaining = max(0.05, float(total_height) - float(top_plinth_height))
        h_step = remaining / float(steps)
    else:
        h_step = float(step_height)
    if h_step <= 0:
        raise ValueError(f"[parts] BASE_STEP_HEIGHT must be > 0 (got {h_step})")

    meshes = []
    z_top = float(col_z_bottom)
    z_cursor = z_top

    # Optional fine steps between column and top plinth.
    # These keep the column position fixed at z_top, but add detail just below it.
    main_top_h = float(top_plinth_height)
    if int(top_steps) > 0:
        main_top_h = max(0.02, main_top_h - float(top_steps) * float(top_step_height))
        r_top_start = max(
            float(column_radius) + 0.01,
            top_r - float(top_steps) * float(top_step_run),
        )
        for i in range(int(top_steps)):
            tier_r = r_top_start + float(i) * float(top_step_run)
            z1 = z_cursor
            z0 = z1 - float(top_step_height)
            meshes.append(_build_box_plinth(tier_r, z0, z1, colour, uv_rect=uv_rect_base))
            z_cursor = z0

    # Top plinth directly under the column.
    z0 = z_cursor - main_top_h
    if top_plinth_texture_rgb is not None and top_plinth_depth_gray is not None:
        meshes.append(
            _build_textured_top_plinth(
                half_extent=top_r,
                z0=z0,
                z1=z_cursor,
                texture_rgb=top_plinth_texture_rgb,
                depth_gray=top_plinth_depth_gray,
                depth_scale=top_plinth_depth_scale,
                u_segments=top_plinth_u_segments,
                v_segments=top_plinth_v_segments,
                base_colour=colour,
                uv_rect=uv_rect_top_plinth,
                flip_v=top_plinth_flip_v,
                depth_invert=top_plinth_depth_invert,
                recess=top_plinth_recess,
                corner_blend_fraction=top_plinth_corner_blend_fraction,
            )
        )
    else:
        meshes.append(_build_box_plinth(top_r, z0, z_cursor, colour, uv_rect=uv_rect_base))
    z_cursor = z0

    # Descending stepped tiers.
    for i in range(int(steps)):
        tier_r = top_r + float(i + 1) * float(step_run)
        z1 = z_cursor
        z0 = z1 - h_step
        meshes.append(_build_box_plinth(tier_r, z0, z1, colour, uv_rect=uv_rect_base))
        z_cursor = z0

    print(
        f"[parts] Base(stepped)  top_r={top_r:.4f}  top_steps={top_steps}  "
        f"top_run={top_step_run:.4f}  top_h={top_step_height:.4f}  "
        f"steps={steps}  run={step_run:.4f}  h_step={h_step:.4f}  "
        f"z=[{z_cursor:.4f}, {z_top:.4f}]"
    )
    return combine_meshes(*meshes)


def build_base(column_radius: float, col_z_bottom: float,
               overhang: float = 0.3, height: float = 0.4,
               style: str = "block",
               top_plinth_height: float = 0.5,
               steps: int = 6,
               step_run: float = 0.25,
               step_height: float = None,
               top_steps: int = 0,
               top_step_run: float = 0.1,
               top_step_height: float = 0.08,
               top_plinth_texture_rgb: np.ndarray = None,
               top_plinth_depth_gray: np.ndarray = None,
               top_plinth_depth_scale: float = 0.03,
               top_plinth_u_segments: int = 128,
               top_plinth_v_segments: int = 16,
               top_plinth_flip_v: bool = False,
               top_plinth_depth_invert: bool = False,
               top_plinth_recess: float = 0.0,
               top_plinth_corner_blend_fraction: float = 0.08,
               uv_rect_base: tuple = (0.0, 0.0, 1.0, 1.0),
               uv_rect_top_plinth: tuple = (0.0, 0.0, 1.0, 1.0),
               colour: tuple = (160, 140, 100)) -> tuple:
    """Base dispatcher. style='block' or style='stepped'."""
    if str(style).strip().lower() == "stepped":
        return build_stepped_base(
            column_radius=column_radius,
            col_z_bottom=col_z_bottom,
            overhang=overhang,
            total_height=height,
            top_plinth_height=top_plinth_height,
            steps=steps,
            step_run=step_run,
            step_height=step_height,
            top_steps=top_steps,
            top_step_run=top_step_run,
            top_step_height=top_step_height,
            top_plinth_texture_rgb=top_plinth_texture_rgb,
            top_plinth_depth_gray=top_plinth_depth_gray,
            top_plinth_depth_scale=top_plinth_depth_scale,
            top_plinth_u_segments=top_plinth_u_segments,
            top_plinth_v_segments=top_plinth_v_segments,
            top_plinth_flip_v=top_plinth_flip_v,
            top_plinth_depth_invert=top_plinth_depth_invert,
            top_plinth_recess=top_plinth_recess,
            top_plinth_corner_blend_fraction=top_plinth_corner_blend_fraction,
            uv_rect_base=uv_rect_base,
            uv_rect_top_plinth=uv_rect_top_plinth,
            colour=colour,
        )
    return build_block_base(
        column_radius=column_radius,
        col_z_bottom=col_z_bottom,
        overhang=overhang,
        height=height,
        uv_rect=uv_rect_base,
        colour=colour,
    )
def build_endcap(radius: float, z_bottom: float, height: float = 0.2,
                 segments: int = 64,
                 colour: tuple = (170, 150, 110)) -> tuple:
    """
    Cylindrical drum sitting on top of the column.
    Spans Z = z_bottom  to  Z = z_bottom + height.
    """
    print(f"[parts] Endcap  r={radius:.4f}  z=[{z_bottom:.4f}, {z_bottom+height:.4f}]")
    # Reuse build_inner_cylinder, just translated up
    v, n, u, c, idx = build_inner_cylinder(radius, height, segments, colour)
    v = v.copy()
    v[:, 2] += z_bottom          # shift up to sit on top of the column
    return v, n, u, c, idx


def _build_wavy_frustum(radius_bottom: float, radius_top: float,
                        z_bottom: float, height: float,
                        segments: int, stacks: int,
                        wave_count: int = 0, wave_amp: float = 0.0,
                        colour: tuple = (170, 150, 110)) -> tuple:
    """
    Frustum shell with optional circumferential wave profile.
    Includes top and bottom caps to keep the part solid.
    """
    if segments < 8 or stacks < 1:
        raise ValueError("[parts] Frustum segments/stacks too low")
    col = np.array(colour, dtype=np.uint8)

    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    ca = np.cos(ang)
    sa = np.sin(ang)
    v_line = np.linspace(0.0, 1.0, stacks + 1, dtype=np.float32)
    aa, vv = np.meshgrid(ang, v_line, indexing="xy")
    ca2 = np.cos(aa)
    sa2 = np.sin(aa)

    base_r = radius_bottom + (radius_top - radius_bottom) * vv
    if wave_count > 0 and abs(wave_amp) > 1e-6:
        wave = np.sin(aa * float(wave_count)) * np.sin(np.pi * vv)
        base_r = base_r + wave_amp * wave
    rr = np.clip(base_r, 1e-4, None)
    zz = z_bottom + height * vv

    verts = np.stack([rr * ca2, rr * sa2, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    normals = np.stack([ca2, sa2, np.zeros_like(ca2)], axis=-1).reshape(-1, 3).astype(np.float32)
    uvs = np.stack([aa / (2.0 * math.pi), vv], axis=-1).reshape(-1, 2).astype(np.float32)
    cols = np.tile(col, (len(verts), 1))

    grid = np.arange((stacks + 1) * segments, dtype=np.int32).reshape(stacks + 1, segments)
    a = grid[:-1, :]
    b = np.roll(a, -1, axis=1)
    c = grid[1:, :]
    d = np.roll(c, -1, axis=1)
    side_idx = np.concatenate(
        [np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1),
         np.stack([c.ravel(), b.ravel(), d.ravel()], axis=1)],
        axis=0,
    ).astype(np.int32)

    # Bottom cap
    r0 = rr[0, :]
    vb = np.stack([r0 * ca, r0 * sa, np.full(segments, z_bottom, dtype=np.float32)], axis=1)
    cb = np.array([[0.0, 0.0, z_bottom]], dtype=np.float32)
    nb = np.tile([0.0, 0.0, -1.0], (segments + 1, 1)).astype(np.float32)
    ub = np.zeros((segments + 1, 2), dtype=np.float32)
    colb = np.tile(col, (segments + 1, 1))
    idx = np.arange(segments, dtype=np.int32)
    nb_idx = np.stack([idx, np.full(segments, segments, dtype=np.int32), np.roll(idx, -1)], axis=1)

    # Top cap
    r1 = rr[-1, :]
    vt = np.stack([r1 * ca, r1 * sa, np.full(segments, z_bottom + height, dtype=np.float32)], axis=1)
    ct = np.array([[0.0, 0.0, z_bottom + height]], dtype=np.float32)
    nt = np.tile([0.0, 0.0, 1.0], (segments + 1, 1)).astype(np.float32)
    ut = np.zeros((segments + 1, 2), dtype=np.float32)
    colt = np.tile(col, (segments + 1, 1))
    nt_idx = np.stack([idx, np.roll(idx, -1), np.full(segments, segments, dtype=np.int32)], axis=1)

    return combine_meshes(
        (verts, normals, uvs, cols, side_idx),
        (np.concatenate([vb, cb], axis=0).astype(np.float32), nb, ub, colb, nb_idx),
        (np.concatenate([vt, ct], axis=0).astype(np.float32), nt, ut, colt, nt_idx),
    )


def build_column_capital(column_radius: float, z_bottom: float,
                         height: float = 1.2,
                         radius_mult: float = 1.65,
                         segments: int = 96,
                         wave_count: int = 10,
                         wave_amp: float = 0.06,
                         abacus_overhang: float = 0.25,
                         abacus_height: float = 0.22,
                         colour: tuple = (170, 150, 110)) -> tuple:
    """
    Approximate classical capital using layered solids:
    neck ring + decorative band + flared echinus + top abacus slab.
    """
    h = float(height)
    r0 = float(column_radius)
    r1 = r0 * float(radius_mult)
    z = float(z_bottom)
    neck_h = max(0.06, h * 0.12)
    band_h = max(0.08, h * 0.20)
    ech_h = max(0.20, h - neck_h - band_h - float(abacus_height))
    slab_h = float(abacus_height)
    print(
        f"[parts] Capital  r0={r0:.4f}  r1={r1:.4f}  z=[{z:.4f}, {z + h:.4f}]  "
        f"h(neck/band/ech/slab)=({neck_h:.3f}/{band_h:.3f}/{ech_h:.3f}/{slab_h:.3f})"
    )

    # Neck ring (cylindrical)
    neck = _build_wavy_frustum(
        radius_bottom=r0 * 1.02,
        radius_top=r0 * 1.06,
        z_bottom=z,
        height=neck_h,
        segments=segments,
        stacks=2,
        wave_count=0,
        wave_amp=0.0,
        colour=colour,
    )

    # Ornamental band (small wave)
    band = _build_wavy_frustum(
        radius_bottom=r0 * 1.06,
        radius_top=r0 * 1.12,
        z_bottom=z + neck_h,
        height=band_h,
        segments=segments,
        stacks=3,
        wave_count=max(6, int(wave_count)),
        wave_amp=wave_amp * 0.45,
        colour=colour,
    )

    # Echinus flare (main profile)
    echinus = _build_wavy_frustum(
        radius_bottom=r0 * 1.12,
        radius_top=r1,
        z_bottom=z + neck_h + band_h,
        height=ech_h,
        segments=segments,
        stacks=6,
        wave_count=max(6, int(wave_count)),
        wave_amp=wave_amp,
        colour=colour,
    )

    # Abacus slab (square top)
    slab_half = r1 + float(abacus_overhang)
    slab = _build_box_plinth(
        half_extent=slab_half,
        z0=z + h - slab_h,
        z1=z + h,
        colour=colour,
    )

    return combine_meshes(neck, band, echinus, slab)


def build_torus_molding(center_radius: float, tube_radius: float, z_center: float,
                        major_segments: int = 96, tube_segments: int = 24,
                        colour: tuple = (185, 165, 125)) -> tuple:
    """
    Build a torus (ring molding) around the column axis.
    center_radius: distance from column axis to torus centerline.
    tube_radius: radius of the torus tube (profile thickness).
    """
    if center_radius <= 0 or tube_radius <= 0:
        raise ValueError(
            f"[parts] Invalid torus dimensions (R={center_radius}, r={tube_radius})"
        )
    if major_segments < 3 or tube_segments < 3:
        raise ValueError("[parts] Torus segments must be >= 3")

    print(
        f"[parts] Molding torus  R={center_radius:.4f}  r={tube_radius:.4f}  "
        f"z={z_center:.4f}  seg=({major_segments},{tube_segments})"
    )
    col = np.array(colour, dtype=np.uint8)

    u = np.linspace(0.0, 2.0 * math.pi, major_segments, endpoint=False, dtype=np.float32)
    v = np.linspace(0.0, 2.0 * math.pi, tube_segments, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    cu = np.cos(uu)
    su = np.sin(uu)
    cv = np.cos(vv)
    sv = np.sin(vv)

    ring = center_radius + tube_radius * cv
    x = ring * cu
    y = ring * su
    z = z_center + tube_radius * sv

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    normals = np.stack([cu * cv, su * cv, sv], axis=-1).reshape(-1, 3).astype(np.float32)
    uvs = np.stack([uu / (2.0 * math.pi), vv / (2.0 * math.pi)], axis=-1).reshape(-1, 2).astype(np.float32)
    cols = np.tile(col, (len(verts), 1))

    grid = np.arange(major_segments * tube_segments, dtype=np.int32).reshape(major_segments, tube_segments)
    a = grid
    b = np.roll(grid, -1, axis=0)
    c = np.roll(grid, -1, axis=1)
    d = np.roll(np.roll(grid, -1, axis=0), -1, axis=1)

    tris = np.concatenate(
        [
            np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1),
            np.stack([c.ravel(), b.ravel(), d.ravel()], axis=1),
        ],
        axis=0,
    ).astype(np.int32)

    return verts, normals, uvs, cols, tris


# ──────────────────────────────────────────────────────────────────────────────
# OBJ / MTL export
# ──────────────────────────────────────────────────────────────────────────────

def export_obj(stem: str, image_path: str,
               verts, normals, uvs, indices):
    """
    Writes <stem>.obj, <stem>.mtl, and copies / references the texture.
    Fully importable in Blender (File → Import → Wavefront OBJ).
    """
    obj_path = stem + ".obj"
    mtl_path = stem + ".mtl"
    tex_name = os.path.basename(image_path)

    print(f"[export] Writing {obj_path} …")
    with open(obj_path, "w") as f:
        f.write(f"# Generated by depth_to_3d.py\n")
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write(f"o DepthMesh\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write("usemtl DepthMeshMat\n")
        f.write("s 1\n")

        # OBJ is 1-indexed; indices already (M, 3)
        for tri in indices:
            i0, i1, i2 = tri[0]+1, tri[1]+1, tri[2]+1
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    with open(mtl_path, "w") as f:
        f.write("# Generated by depth_to_3d.py\n")
        f.write("newmtl DepthMeshMat\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {tex_name}\n")

    print(f"[export] Done → {obj_path}  +  {mtl_path}")
    print(f"[export] Make sure '{tex_name}' is in the same folder when importing into Blender.")


def export_glb(stem: str, image_path: str,
               verts, normals, uvs, indices, colour: np.ndarray):
    """Export as GLB using trimesh (pip install trimesh)."""
    try:
        import trimesh
        from trimesh.visual.texture import TextureVisuals
        from trimesh.visual.material import PBRMaterial
    except ImportError:
        print("[export] trimesh not installed – skipping GLB export.")
        print("         Run:  pip install trimesh[easy]")
        return

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=indices,
        vertex_normals=normals,
    )

    # Attach texture
    img = Image.open(image_path).convert("RGBA")
    material = PBRMaterial(baseColorTexture=img)
    mesh.visual = TextureVisuals(uv=uvs, material=material)

    glb_path = stem + ".glb"
    mesh.export(glb_path)
    print(f"[export] Done → {glb_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Pyglet viewer
# ──────────────────────────────────────────────────────────────────────────────

def run_viewer(verts, normals, uvs, indices, colour: np.ndarray,
               image_path: str, export_fmt: str, out_stem: str,
               col_height: float = 0.0,
               lip_tri_mask: np.ndarray = None,
               statue_tri_mask: np.ndarray = None,
               statue_tex_rgb: np.ndarray = None):
    """
    Opens an interactive pyglet window showing the 3-D mesh.
    Uses OpenGL immediate-mode via pyglet's low-level GL bindings so that
    no external shader library is required.
    col_height > 0 switches the camera to column mode (looking at a tall object).
    """
    try:
        import pyglet
        from pyglet.gl import (
            glEnable, glDisable, glClearColor, glClear, glLoadIdentity,
            glMatrixMode, glLoadMatrixf, glOrtho, glFrustum,
            glTranslatef, glRotatef, glScalef,
            glColor4f,
            glEnableClientState, glDisableClientState,
            glVertexPointer, glNormalPointer, glTexCoordPointer, glColorPointer,
            glDrawElements,
            glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
            glPixelStorei,
            glGetIntegerv,
            glLightfv, glMaterialfv, glColorMaterial,
            GL_TRIANGLES, GL_UNSIGNED_INT,
            GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_TEXTURE_COORD_ARRAY,
            GL_FLOAT, GL_UNSIGNED_BYTE,
            GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
            GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_TEXTURE_2D,
            GL_PROJECTION, GL_MODELVIEW,
            GL_AMBIENT_AND_DIFFUSE, GL_FRONT_AND_BACK,
            GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR,
            GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE,
            GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
            GL_UNPACK_ALIGNMENT,
            GL_MAX_TEXTURE_SIZE,
            GL_COLOR_MATERIAL,
            glGenerateMipmap,
        )
        from pyglet.gl import GLfloat, GLuint, GLint
    except ImportError:
        print("[viewer] pyglet not installed – skipping viewer.")
        print("         Run:  pip install pyglet")
        return

    # ── flatten arrays for OpenGL ──────────────────────────────────────────
    verts_f   = verts.astype(np.float32).ravel()
    normals_f = normals.astype(np.float32).ravel()
    uvs_f     = uvs.astype(np.float32).ravel()
    lip_mask = lip_tri_mask if (lip_tri_mask is not None and len(lip_tri_mask) == len(indices)) else np.zeros(len(indices), dtype=bool)
    statue_mask = statue_tri_mask if (statue_tri_mask is not None and len(statue_tri_mask) == len(indices)) else np.zeros(len(indices), dtype=bool)
    main_mask = ~(lip_mask | statue_mask)

    main_indices = indices[main_mask].astype(np.uint32)
    lip_indices = indices[lip_mask].astype(np.uint32)
    statue_indices = indices[statue_mask].astype(np.uint32)

    idx_flat_main = main_indices.ravel()
    idx_flat_lip = lip_indices.ravel()
    idx_flat_statue = statue_indices.ravel()
    n_idx_main = len(idx_flat_main)
    n_idx_lip = len(idx_flat_lip)
    n_idx_statue = len(idx_flat_statue)

    # ── load texture ───────────────────────────────────────────────────────
    tex_img_src = Image.open(image_path).convert("RGB")

    # ── camera state ───────────────────────────────────────────────────────
    # For a column, default to a front-on full-height view.
    if col_height > 0:
        init_dist  = max(2.0, col_height * float(VIEW_COLUMN_DIST_FACTOR))
        init_pitch = float(VIEW_COLUMN_INIT_PITCH)
        init_yaw   = float(VIEW_COLUMN_INIT_YAW)
        init_pan_y = (-col_height * 0.5) if VIEW_COLUMN_CENTER_HEIGHT else 0.0
    else:
        init_dist  = 3.5
        init_pitch = 20.0
        init_yaw   = 0.0
        init_pan_y = 0.0

    cam = {
        "yaw":   init_yaw,
        "pitch": init_pitch,
        "spin_z": 0.0,
        "dist":  init_dist,
        "pan_x": 0.0,
        "pan_y": init_pan_y,
        "drag_btn": None,
        "last_x": 0,
        "last_y": 0,
        "init_yaw":   init_yaw,
        "init_dist":  init_dist,
        "init_pitch": init_pitch,
        "init_pan_y": init_pan_y,
    }
    auto_rotate = {"enabled": False, "speed_deg_s": 20.0}

    config = pyglet.gl.Config(double_buffer=True, depth_size=24, samples=4)
    try:
        window = pyglet.window.Window(
            width=1280, height=720,
            caption="Joshua Roll Column Simulator  |  drag=orbit  right-drag=pan  scroll=zoom  A=auto-rotate  E=export  R=reset  Q=quit",
            resizable=True, config=config)
    except Exception:
        window = pyglet.window.Window(
            width=1280, height=720,
            caption="Joshua Roll Column Simulator",
            resizable=True)

    # ── GL state (initialised lazily on first draw, inside the GL context) ──
    gl_state = {"ready": False, "tex_id_main": None, "tex_id_lip": None, "tex_id_statue": None,
                "vp": None, "np_": None, "tp": None,
                "ip_main": None, "ip_lip": None, "ip_statue": None}

    def gl_init():
        """Upload texture and cache ctypes pointers. Called once inside on_draw."""
        max_tex = GLint(0)
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, ctypes.byref(max_tex))
        max_tex_size = int(max_tex.value) if int(max_tex.value) > 0 else 4096

        tex_img = tex_img_src
        tw, th = tex_img.size
        if tw > max_tex_size or th > max_tex_size:
            scale = min(max_tex_size / float(tw), max_tex_size / float(th))
            nw = max(1, int(round(tw * scale)))
            nh = max(1, int(round(th * scale)))
            print(f"[viewer] Resizing preview texture {tw}x{th} -> {nw}x{nh} (GL max {max_tex_size})")
            tex_img = tex_img.resize((nw, nh), Image.LANCZOS)
            tw, th = nw, nh
        else:
            print(f"[viewer] Texture upload size: {tw}x{th} (GL max {max_tex_size})")

        lip_tex_img = tex_img.filter(ImageFilter.GaussianBlur(radius=LIP_TEXTURE_BLUR_RADIUS))
        tex_data = np.array(tex_img, dtype=np.uint8).ravel()
        lip_tex_data = np.array(lip_tex_img, dtype=np.uint8).ravel()

        tex_id_main = GLuint(0)
        glGenTextures(1, ctypes.byref(tex_id_main))
        glBindTexture(GL_TEXTURE_2D, tex_id_main)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tw, th, 0,
                     GL_RGB, GL_UNSIGNED_BYTE,
                     tex_data.ctypes.data_as(ctypes.c_void_p))
        try:
            glGenerateMipmap(GL_TEXTURE_2D)
        except Exception:
            # Fallback for contexts that reject mipmap generation for this texture.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        tex_id_lip = GLuint(0)
        glGenTextures(1, ctypes.byref(tex_id_lip))
        glBindTexture(GL_TEXTURE_2D, tex_id_lip)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tw, th, 0,
                     GL_RGB, GL_UNSIGNED_BYTE,
                     lip_tex_data.ctypes.data_as(ctypes.c_void_p))
        try:
            glGenerateMipmap(GL_TEXTURE_2D)
        except Exception:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        tex_id_statue = None
        if statue_tex_rgb is not None:
            st_h, st_w = statue_tex_rgb.shape[0], statue_tex_rgb.shape[1]
            if st_w > max_tex_size or st_h > max_tex_size:
                scale = min(max_tex_size / float(st_w), max_tex_size / float(st_h))
                nw = max(1, int(round(st_w * scale)))
                nh = max(1, int(round(st_h * scale)))
                st_img = Image.fromarray(statue_tex_rgb, mode="RGB").resize((nw, nh), Image.LANCZOS)
                st_data = np.array(st_img, dtype=np.uint8).ravel()
                st_w, st_h = nw, nh
            else:
                st_data = statue_tex_rgb.astype(np.uint8).ravel()

            tex_id_statue = GLuint(0)
            glGenTextures(1, ctypes.byref(tex_id_statue))
            glBindTexture(GL_TEXTURE_2D, tex_id_statue)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, st_w, st_h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE,
                         st_data.ctypes.data_as(ctypes.c_void_p))
            try:
                glGenerateMipmap(GL_TEXTURE_2D)
            except Exception:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        gl_state["tex_id_main"] = tex_id_main
        gl_state["tex_id_lip"] = tex_id_lip
        gl_state["tex_id_statue"] = tex_id_statue
        gl_state["vp"]  = verts_f.ctypes.data_as(ctypes.c_void_p)
        gl_state["np_"] = normals_f.ctypes.data_as(ctypes.c_void_p)
        gl_state["tp"]  = uvs_f.ctypes.data_as(ctypes.c_void_p)
        gl_state["ip_main"] = idx_flat_main.ctypes.data_as(ctypes.c_void_p)
        gl_state["ip_lip"] = idx_flat_lip.ctypes.data_as(ctypes.c_void_p) if n_idx_lip > 0 else None
        gl_state["ip_statue"] = idx_flat_statue.ctypes.data_as(ctypes.c_void_p) if n_idx_statue > 0 else None
        gl_state["ready"] = True

    def reset_camera():
        cam["yaw"]   = cam["init_yaw"]
        cam["pitch"] = cam["init_pitch"]
        cam["spin_z"] = 0.0
        cam["dist"]  = cam["init_dist"]
        cam["pan_x"] = 0.0
        cam["pan_y"] = cam["init_pan_y"]

    @window.event
    def on_draw():
        # Initialise GL resources on the very first draw call (context is live here)
        if not gl_state["ready"]:
            gl_init()

        window.clear()
        glClearColor(0.15, 0.15, 0.18, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ── projection ────────────────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w, h = window.width, window.height
        aspect = w / max(h, 1)
        far_plane = max(100.0, col_height * 20) if col_height > 0 else 100.0
        near, far = 0.01, far_plane
        fov_y = 45.0
        f = 1.0 / math.tan(math.radians(fov_y) / 2)
        proj = (GLfloat * 16)(
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), -1,
            0, 0, (2*far*near)/(near-far), 0,
        )
        glLoadMatrixf(proj)

        # ── modelview ─────────────────────────────────────────────────────
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(cam["pan_x"], cam["pan_y"], -cam["dist"])
        glRotatef(cam["pitch"], 1, 0, 0)
        glRotatef(cam["yaw"],   0, 1, 0)
        glRotatef(cam["spin_z"], 0, 0, 1)

        # ── lighting ──────────────────────────────────────────────────────
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        light_pos = (GLfloat * 4)(*LIGHT_POSITION)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        light_diff = (GLfloat * 4)(*LIGHT_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diff)
        light_amb  = (GLfloat * 4)(*LIGHT_AMBIENT)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb)

        # ── texture ───────────────────────────────────────────────────────
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, gl_state["tex_id_main"])

        # ── draw ──────────────────────────────────────────────────────────
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, gl_state["vp"])
        glNormalPointer(GL_FLOAT, 0, gl_state["np_"])
        glTexCoordPointer(2, GL_FLOAT, 0, gl_state["tp"])

        if n_idx_main > 0:
            glDrawElements(GL_TRIANGLES, n_idx_main, GL_UNSIGNED_INT, gl_state["ip_main"])

        if n_idx_lip > 0 and gl_state["ip_lip"] is not None:
            glBindTexture(GL_TEXTURE_2D, gl_state["tex_id_lip"])
            glDrawElements(GL_TRIANGLES, n_idx_lip, GL_UNSIGNED_INT, gl_state["ip_lip"])

        if n_idx_statue > 0 and gl_state["ip_statue"] is not None:
            if gl_state["tex_id_statue"] is not None:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, gl_state["tex_id_statue"])
            else:
                glDisable(GL_TEXTURE_2D)
                glColor4f(0.69, 0.49, 0.29, 1.0)
            glDrawElements(GL_TRIANGLES, n_idx_statue, GL_UNSIGNED_INT, gl_state["ip_statue"])

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        cam["drag_btn"] = button
        cam["last_x"] = x
        cam["last_y"] = y

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        cam["drag_btn"] = None

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        from pyglet.window import mouse
        if buttons & mouse.LEFT:
            cam["yaw"]   += dx * 0.4
            cam["pitch"] -= dy * 0.4
            cam["pitch"]  = max(-89, min(89, cam["pitch"]))
        elif buttons & mouse.RIGHT:
            cam["pan_x"] += dx * 0.005 * cam["dist"]
            cam["pan_y"] += dy * 0.005 * cam["dist"]

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        cam["dist"] -= scroll_y * 0.15 * cam["dist"]
        max_dist = max(50.0, col_height * 10) if col_height > 0 else 50.0
        cam["dist"]  = max(0.1, min(max_dist, cam["dist"]))

    @window.event
    def on_key_press(symbol, modifiers):
        from pyglet.window import key
        if symbol in (key.Q, key.ESCAPE):
            window.close()
        elif symbol == key.R:
            reset_camera()
        elif symbol == key.A:
            auto_rotate["enabled"] = not auto_rotate["enabled"]
            state = "ON" if auto_rotate["enabled"] else "OFF"
            print(f"[viewer] Auto-rotate: {state}")
        elif symbol == key.E:
            if export_fmt == "obj":
                export_obj(out_stem, image_path, verts, normals, uvs, indices)
            elif export_fmt == "glb":
                export_glb(out_stem, image_path, verts, normals, uvs, indices, colour)
            elif export_fmt == "both":
                export_obj(out_stem, image_path, verts, normals, uvs, indices)
                export_glb(out_stem, image_path, verts, normals, uvs, indices, colour)
            else:
                print("[export] Export is disabled (EXPORT = 'none').")

    @window.event
    def on_resize(width, height):
        from pyglet.gl import glViewport
        glViewport(0, 0, width, height)

    print("[viewer] Opening 3D viewer …  (E=export  R=reset  Q/Esc=quit)")
    # In some IDE run modes, stale exit state can leak between runs.
    window.has_exit = False
    draw_error_reported = False
    prev_t = time.perf_counter()
    while not window.has_exit:
        now_t = time.perf_counter()
        dt = now_t - prev_t
        prev_t = now_t
        if auto_rotate["enabled"]:
            # Spin around model Z (vertical symmetry axis).
            cam["spin_z"] += auto_rotate["speed_deg_s"] * dt

        window.switch_to()
        window.dispatch_events()
        try:
            window.dispatch_event("on_draw")
            window.flip()
        except Exception:
            if not draw_error_reported:
                import traceback
                print("[viewer] Error during draw; keeping window open for debugging:")
                traceback.print_exc()
                draw_error_reported = True
            try:
                window.clear()
                window.flip()
            except Exception:
                pass
        pyglet.clock.tick()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── validate paths ────────────────────────────────────────────────────
    for p in (IMAGE, DEPTH):
        if not os.path.isfile(p):
            sys.exit(f"[error] File not found: {p}")

    print("=" * 60)
    print("  depth_to_3d.py")
    print("=" * 60)
    print(f"  image           : {IMAGE}")
    print(f"  depth map       : {DEPTH}")
    print(f"  contrast        : {CONTRAST}")
    print(f"  depth_threshold : {DEPTH_THRESHOLD}")
    print(f"  blur_radius     : {BLUR_RADIUS}")
    print(f"  depth_scale     : {DEPTH_SCALE}")
    print(f"  backing_depth   : {BACKING_DEPTH}")
    print(f"  wrap_column     : {WRAP_COLUMN}")
    print(f"  helix_angle     : {HELIX_ANGLE_DEG}°")
    print(f"  column_radius   : {'auto' if COLUMN_RADIUS is None else COLUMN_RADIUS}")
    print(f"  inner_cylinder  : {ADD_INNER_CYLINDER}  (seg={INNER_CYLINDER_SEGMENTS})")
    print(f"  lip             : {ADD_LIP}  (h={'auto' if LIP_HEIGHT is None else LIP_HEIGHT}  depth={LIP_DEPTH})")
    print(f"  lip_tex_blur    : {LIP_TEXTURE_BLUR_RADIUS}px  (strip={LIP_TEXTURE_STRIP_FRACTION})")
    print(f"  end_fade        : {END_FADE_ENABLED}  (frac={END_FADE_FRACTION}  power={END_FADE_POWER})")
    print(f"  mirror_edges    : {MIRROR_EDGE_ENABLED}  (frac={MIRROR_EDGE_FRACTION})")
    print(f"  top_clip        : {CLIP_HELIX_AT_ENDCAP}  (eps={HELIX_TOP_CLIP_EPS})")
    print(f"  moldings        : {ADD_MOLDINGS}  (bottom={ADD_BOTTOM_MOLDING}  top={ADD_TOP_MOLDING})")
    print(f"  molding params  : roff={MOLDING_RADIUS_OFFSET}  tube={MOLDING_TUBE_RADIUS}  seg=({MOLDING_MAJOR_SEGMENTS},{MOLDING_TUBE_SEGMENTS})")
    print(
        f"  base            : {ADD_BASE}  (style={BASE_STYLE}  overhang={BASE_OVERHANG}  h={BASE_HEIGHT}  "
        f"top_h={BASE_TOP_PLINTH_HEIGHT}  steps={BASE_STEPS}  run={BASE_STEP_RUN}  step_h={BASE_STEP_HEIGHT}  "
        f"top_steps={BASE_TOP_STEPS}  top_run={BASE_TOP_STEP_RUN}  top_step_h={BASE_TOP_STEP_HEIGHT})"
    )
    print(
        f"  plinth tex/depth: {PLINTH_SIDE_IMAGE} / {PLINTH_SIDE_DEPTH}  "
        f"(scale={PLINTH_SIDE_DEPTH_SCALE}  flip_v={PLINTH_SIDE_FLIP_V}  invert={PLINTH_SIDE_DEPTH_INVERT}  "
        f"recess={PLINTH_SIDE_RECESS}  corner_blend={PLINTH_CORNER_BLEND_FRACTION}  "
        f"step_blur={PLINTH_STEP_TEXTURE_BLUR_RADIUS})"
    )
    print(f"  endcap          : {ADD_ENDCAP}  (h={ENDCAP_HEIGHT})")
    print(
        f"  capital         : {ADD_CAPITAL}  (h={CAPITAL_HEIGHT}  r_mult={CAPITAL_RADIUS_MULT}  "
        f"waves={CAPITAL_WAVE_COUNT}  amp={CAPITAL_WAVE_AMPLITUDE}  tex={CAPITAL_TEXTURE_IMAGE}  "
        f"blur={CAPITAL_TEXTURE_BLUR_RADIUS}  per_side={CAPITAL_TEXTURE_WRAP_PER_SIDE})"
    )
    print(
        f"  top_statue      : {ADD_TOP_STATUE}  (glb={TOP_STATUE_GLB}  "
        f"h_mult={TOP_STATUE_HEIGHT_MULT}  z_gap={TOP_STATUE_Z_GAP})"
    )
    print(f"  view(column)    : yaw={VIEW_COLUMN_INIT_YAW}  pitch={VIEW_COLUMN_INIT_PITCH}  distx={VIEW_COLUMN_DIST_FACTOR}")
    print(f"  light_position  : {LIGHT_POSITION}")
    print(f"  auto_part_color : {AUTO_MATCH_PART_COLORS}")
    print(f"  downsample      : {DOWNSAMPLE}")
    print(f"  export          : {EXPORT}")
    print(f"  output stem     : {OUT}")
    print("=" * 60)

    # ── preprocessing ─────────────────────────────────────────────────────
    colour, depth = load_and_preprocess(
        IMAGE, DEPTH,
        contrast=CONTRAST,
        downsample=DOWNSAMPLE,
        depth_threshold=DEPTH_THRESHOLD,
        blur_radius=BLUR_RADIUS,
    )

    if AUTO_MATCH_PART_COLORS:
        part_base_rgb, _part_lip_rgb, part_accent_rgb = derive_part_palette_from_texture(IMAGE)
    else:
        part_base_rgb, _part_lip_rgb, part_accent_rgb = (180, 160, 120), (175, 155, 115), (170, 150, 110)
    print(f"  part colors     : base={part_base_rgb}  accent={part_accent_rgb}")

    # Optional plinth texture/depth setup + dedicated base color from plinth image.
    render_image_path = IMAGE
    uv_rect_main = (0.0, 0.0, 1.0, 1.0)
    uv_rect_base = (0.0, 0.0, 1.0, 1.0)
    uv_rect_plinth = (0.0, 0.0, 1.0, 1.0)
    uv_rect_plinth_blur = (0.0, 0.0, 1.0, 1.0)
    uv_rect_capital = (0.0, 0.0, 1.0, 1.0)
    plinth_tex_rgb = None
    plinth_depth_gray = None
    base_part_rgb = part_accent_rgb
    statue_tex_rgb = None

    if os.path.isfile(PLINTH_SIDE_IMAGE):
        base_part_rgb = average_rgb_from_image(PLINTH_SIDE_IMAGE)
        print(f"  base color avg  : {base_part_rgb}  (from {PLINTH_SIDE_IMAGE})")

        atlas_path = f"{OUT}_atlas.png"
        capital_tex_path = CAPITAL_TEXTURE_IMAGE if os.path.isfile(CAPITAL_TEXTURE_IMAGE) else None
        render_image_path, uv_rect_main, uv_rect_plinth, uv_rect_plinth_blur, uv_rect_capital = build_texture_atlas_horizontal(
            IMAGE, PLINTH_SIDE_IMAGE, atlas_path,
            extra_blur_radius=PLINTH_STEP_TEXTURE_BLUR_RADIUS,
            extra2_image_path=capital_tex_path,
            extra2_blur_radius=CAPITAL_TEXTURE_BLUR_RADIUS,
        )
        uv_rect_base = uv_rect_plinth_blur
        print(f"[tex] Built atlas: {render_image_path}")
        print(
            f"[tex] UV rects  main={uv_rect_main}  plinth={uv_rect_plinth}  "
            f"plinth_blur={uv_rect_plinth_blur}  capital={uv_rect_capital}"
        )
        if capital_tex_path is None:
            print(f"[warn] Capital texture not found: {CAPITAL_TEXTURE_IMAGE} (capital uses main texture)")

        if os.path.isfile(PLINTH_SIDE_DEPTH):
            plinth_tex_rgb = np.asarray(Image.open(PLINTH_SIDE_IMAGE).convert("RGB"), dtype=np.uint8)
            plinth_depth_gray = np.asarray(Image.open(PLINTH_SIDE_DEPTH).convert("L"), dtype=np.float32) / 255.0
        else:
            print(f"[warn] Missing plinth depth map: {PLINTH_SIDE_DEPTH} (top plinth relief disabled)")
    else:
        print(f"[warn] Missing plinth texture: {PLINTH_SIDE_IMAGE} (using default base color)")
        if os.path.isfile(CAPITAL_TEXTURE_IMAGE):
            atlas_path = f"{OUT}_atlas.png"
            render_image_path, uv_rect_main, _tmp_rect, _tmp_blur_rect, uv_rect_capital = build_texture_atlas_horizontal(
                IMAGE, CAPITAL_TEXTURE_IMAGE, atlas_path,
                extra_blur_radius=0.0,
                extra2_image_path=None,
                extra2_blur_radius=0.0,
            )
            # For this fallback atlas, extra rect corresponds to capital texture.
            uv_rect_capital = _tmp_rect
            print(f"[tex] Built atlas (main+capital): {render_image_path}")
            print(f"[tex] UV rects  main={uv_rect_main}  capital={uv_rect_capital}")

    # ── resolve shared geometry parameters ───────────────────────────────
    R = 1.0 if COLUMN_RADIUS is None else float(COLUMN_RADIUS)

    # ── part 1: helix relief strip ────────────────────────────────────────
    print("[mesh] Building helix relief strip …")
    verts, normals, uvs, cols, indices, col_height, lip_tri_mask = build_mesh(
        colour, depth, DEPTH_SCALE,
        backing_depth=BACKING_DEPTH,
        wrap_column=WRAP_COLUMN,
        helix_angle_deg=HELIX_ANGLE_DEG,
        column_radius=R,
        add_lip=ADD_LIP,
        lip_height=LIP_HEIGHT,
        lip_depth=LIP_DEPTH,
        end_fade_enabled=END_FADE_ENABLED,
        end_fade_fraction=END_FADE_FRACTION,
        end_fade_power=END_FADE_POWER,
        mirror_edge_enabled=MIRROR_EDGE_ENABLED,
        mirror_edge_fraction=MIRROR_EDGE_FRACTION,
    )
    uvs = map_uv_rect(uvs, uv_rect_main)
    parts = [(verts, normals, uvs, cols, indices)]
    statue_tri_count = 0

    if WRAP_COLUMN:
        # ── part 2: inner solid cylinder ──────────────────────────────────
        if ADD_INNER_CYLINDER:
            cyl_v, cyl_n, cyl_u, cyl_c, cyl_i = build_inner_cylinder(
                radius=R - BACKING_DEPTH,   # fits just inside the relief shell
                height=col_height,
                segments=INNER_CYLINDER_SEGMENTS,
                colour=part_base_rgb,
            )
            cyl = (cyl_v, cyl_n, map_uv_rect(cyl_u, uv_rect_main), cyl_c, cyl_i)
            parts.append(cyl)

        # ── part 3: rectangular base / plinth ─────────────────────────────
        if ADD_BASE:
            base = build_base(
                column_radius=R,
                col_z_bottom=0.0,           # column base sits at Z=0
                overhang=BASE_OVERHANG,
                height=BASE_HEIGHT,
                style=BASE_STYLE,
                top_plinth_height=BASE_TOP_PLINTH_HEIGHT,
                steps=BASE_STEPS,
                step_run=BASE_STEP_RUN,
                step_height=BASE_STEP_HEIGHT,
                top_steps=BASE_TOP_STEPS,
                top_step_run=BASE_TOP_STEP_RUN,
                top_step_height=BASE_TOP_STEP_HEIGHT,
                top_plinth_texture_rgb=plinth_tex_rgb,
                top_plinth_depth_gray=plinth_depth_gray,
                top_plinth_depth_scale=PLINTH_SIDE_DEPTH_SCALE,
                top_plinth_u_segments=PLINTH_SIDE_U_SEGMENTS,
                top_plinth_v_segments=PLINTH_SIDE_V_SEGMENTS,
                top_plinth_flip_v=PLINTH_SIDE_FLIP_V,
                top_plinth_depth_invert=PLINTH_SIDE_DEPTH_INVERT,
                top_plinth_recess=PLINTH_SIDE_RECESS,
                top_plinth_corner_blend_fraction=PLINTH_CORNER_BLEND_FRACTION,
                uv_rect_base=uv_rect_base,
                uv_rect_top_plinth=uv_rect_plinth,
                colour=base_part_rgb,
            )
            parts.append(base)

        # ── part 4: endcap drum on top ────────────────────────────────────
        if ADD_ENDCAP:
            ec_r = R if ENDCAP_RADIUS is None else float(ENDCAP_RADIUS)
            cap_v, cap_n, cap_u, cap_c, cap_i = build_endcap(
                radius=ec_r,
                z_bottom=col_height,
                height=ENDCAP_HEIGHT,
                segments=ENDCAP_SEGMENTS,
                colour=part_base_rgb,
            )
            cap = (cap_v, cap_n, map_uv_rect(cap_u, uv_rect_main), cap_c, cap_i)
            parts.append(cap)

        # Optional classical capital approximation.
        capital_top_z = col_height
        if ADD_CAPITAL:
            capl_v, capl_n, capl_u, capl_c, capl_i = build_column_capital(
                column_radius=R,
                z_bottom=col_height,
                height=CAPITAL_HEIGHT,
                radius_mult=CAPITAL_RADIUS_MULT,
                segments=CAPITAL_SEGMENTS,
                wave_count=CAPITAL_WAVE_COUNT,
                wave_amp=CAPITAL_WAVE_AMPLITUDE,
                abacus_overhang=CAPITAL_ABACUS_OVERHANG,
                abacus_height=CAPITAL_ABACUS_HEIGHT,
                colour=base_part_rgb,
            )
            capital = (
                capl_v,
                capl_n,
                map_capital_side_uv(
                    capl_v, capl_u, uv_rect_capital,
                    flip_v=CAPITAL_TEXTURE_FLIP_V,
                    wrap_per_side=CAPITAL_TEXTURE_WRAP_PER_SIDE,
                ),
                capl_c,
                capl_i,
            )
            parts.append(capital)
            capital_top_z = col_height + float(CAPITAL_HEIGHT)

        # Torus/fillet moldings.
        if ADD_MOLDINGS:
            mold_r = R + float(MOLDING_RADIUS_OFFSET)
            mold_t = float(MOLDING_TUBE_RADIUS)
            if ADD_BOTTOM_MOLDING:
                bm_v, bm_n, bm_u, bm_c, bm_i = build_torus_molding(
                    center_radius=mold_r,
                    tube_radius=mold_t,
                    z_center=float(MOLDING_BOTTOM_Z),
                    major_segments=MOLDING_MAJOR_SEGMENTS,
                    tube_segments=MOLDING_TUBE_SEGMENTS,
                    colour=base_part_rgb,
                )
                bottom_mold = (bm_v, bm_n, map_uv_rect(bm_u, uv_rect_base), bm_c, bm_i)
                parts.append(bottom_mold)
            if ADD_TOP_MOLDING:
                tm_v, tm_n, tm_u, tm_c, tm_i = build_torus_molding(
                    center_radius=mold_r,
                    tube_radius=mold_t,
                    z_center=col_height + float(MOLDING_TOP_Z_OFFSET),
                    major_segments=MOLDING_MAJOR_SEGMENTS,
                    tube_segments=MOLDING_TUBE_SEGMENTS,
                    colour=base_part_rgb,
                )
                top_mold = (tm_v, tm_n, map_uv_rect(tm_u, uv_rect_main), tm_c, tm_i)
                parts.append(top_mold)

        # Optional external statue (OBJ) placed above the capital.
        # Appended last so we can isolate it for bronze rendering in viewer.
        if ADD_TOP_STATUE:
            if os.path.isfile(TOP_STATUE_GLB):
                try:
                    sv, sn, su, sc, si, statue_tex_rgb = load_glb_mesh(TOP_STATUE_GLB, colour=base_part_rgb)
                    if TOP_STATUE_TEXTURE_FLIP_V:
                        su = su.copy().astype(np.float32)
                        su[:, 1] = 1.0 - su[:, 1]

                    # Re-orient imported statue (common up-axis mismatch).
                    # +90 deg around X: (x, y, z) -> (x, -z, y)
                    sv = np.stack([sv[:, 0], -sv[:, 2], sv[:, 1]], axis=1).astype(np.float32)

                    bb_min = np.min(sv, axis=0)
                    bb_max = np.max(sv, axis=0)
                    src_h = float(bb_max[2] - bb_min[2])
                    if src_h <= 1e-8:
                        raise ValueError(f"[parts] Statue height is zero in {TOP_STATUE_GLB}")

                    target_h = float(CAPITAL_HEIGHT) * float(TOP_STATUE_HEIGHT_MULT)
                    s = target_h / src_h
                    sv = sv * s

                    bb_min = np.min(sv, axis=0)
                    bb_max = np.max(sv, axis=0)
                    cx = 0.5 * float(bb_min[0] + bb_max[0])
                    cy = 0.5 * float(bb_min[1] + bb_max[1])
                    zmin = float(bb_min[2])

                    sv[:, 0] -= cx
                    sv[:, 1] -= cy
                    sv[:, 2] -= zmin
                    sv[:, 2] += capital_top_z + float(TOP_STATUE_Z_GAP)

                    statue = (sv, sn, su.astype(np.float32), sc, si)
                    parts.append(statue)
                    statue_tri_count = len(si)
                    print(
                        f"[parts] Top statue  file={TOP_STATUE_GLB}  "
                        f"src_h={src_h:.4f}  target_h={target_h:.4f}  scale={s:.4f}  "
                        f"z_base={capital_top_z + float(TOP_STATUE_Z_GAP):.4f}"
                    )
                except Exception as e:
                    print(f"[warn] Failed to place top statue '{TOP_STATUE_GLB}': {e}")
            else:
                print(f"[warn] Top statue GLB not found: {TOP_STATUE_GLB}")

    # ── combine all parts into one mesh ───────────────────────────────────
    statue_tri_mask = np.zeros(len(indices), dtype=bool)
    if len(parts) > 1:
        base_tri_count = len(indices)
        verts, normals, uvs, cols, indices = combine_meshes(*parts)
        extra_tri_count = len(indices) - base_tri_count
        if extra_tri_count > 0:
            lip_tri_mask = np.concatenate(
                [lip_tri_mask, np.zeros(extra_tri_count, dtype=bool)],
                axis=0,
            )
        if statue_tri_count > 0 and statue_tri_count <= len(indices):
            statue_tri_mask = np.zeros(len(indices), dtype=bool)
            statue_tri_mask[-statue_tri_count:] = True
        print(f"[mesh] Combined: {len(verts):,} vertices  |  {len(indices):,} triangles")

    # Validate index bounds to prevent driver-level crashes in the viewer.
    i_min = int(indices.min())
    i_max = int(indices.max())
    if i_min < 0 or i_max >= len(verts):
        raise ValueError(
            f"[mesh] Invalid index range: min={i_min}, max={i_max}, vertex_count={len(verts)}"
        )

    # ── viewer (press E inside the window to export) ──────────────────────
    run_viewer(verts, normals, uvs, indices, colour, render_image_path,
               export_fmt=EXPORT, out_stem=OUT,
               col_height=col_height if WRAP_COLUMN else 0.0,
               lip_tri_mask=lip_tri_mask,
               statue_tri_mask=statue_tri_mask,
               statue_tex_rgb=statue_tex_rgb)


if __name__ == "__main__":
    main()
