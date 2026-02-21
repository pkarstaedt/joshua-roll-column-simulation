# Joshua Roll Column Simulation

A Python-based simulation that maps the Joshua Roll manuscript onto a 3D helical triumphal column (similar to Trajan's Column).

## Step 1: Helical Model Generation
The initial phase focuses on the precise mathematical mapping of the long, horizontal strip of the manuscript onto a vertical cylinder.

### Key Achievements:
- **Precise Helical Mapping:** Calculated the column radius ($R \approx 492$ units) and height ($H \approx 8838$ units) based on a fixed 15-degree helix angle to ensure a perfect, seamless wrap of the `jc_roll_small.jpg` texture.
- **Dynamic Proportions:** The model automatically adjusts its geometry based on the input image dimensions and the specified wrap angle.
- **Pyglet 3D Viewer:** Implemented an interactive OpenGL-based renderer with support for:
  - **Rotation:** Click and drag to rotate the column.
  - **Vertical Panning:** Right-click drag or Shift-drag to scan up and down the height of the column.
  - **Zooming:** Scroll wheel to inspect fine details.
  - **White Background:** Clean presentation for archaeological visualization.
- **Large Texture Support:** Integrated PIL (Pillow) to handle high-resolution manuscript images (31,160 pixels wide) and corrected UV mapping to account for power-of-two padding in OpenGL.
- **Export Functionality:** Added the ability to export the generated 3D model to `.obj` and `.mtl` formats for use in professional 3D software (Blender, Maya, etc.).

### Usage:
1. Ensure dependencies are installed: `pip install pyglet Pillow`
2. Run the simulation: `python helical_column_fixed.py`
3. Press **'E'** in the viewer to export the model as `joshua_roll_column.obj`.

## Current Status:
Successfully achieved the basic "Triumphal Column" structure with accurate texture application and interactive viewing.
