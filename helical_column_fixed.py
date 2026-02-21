import pyglet
from pyglet.gl import *
from pyglet.gl.glu import *
import math
from PIL import Image

# Load texture info
texture_file = 'jc_roll_small.jpg'
try:
    with Image.open(texture_file) as img:
        L, W_strip = img.size
except FileNotFoundError:
    print(f"Error: {texture_file} not found.")
    exit(1)

# Parameters
angle_deg = 15.0
angle_rad = math.radians(angle_deg)

# Calculate radius so that edges touch perfectly: 2*pi*R*sin(alpha) = W_strip
R = W_strip / (2 * math.pi * math.sin(angle_rad))
H = L * math.sin(angle_rad) + W_strip * math.cos(angle_rad)

print(f"Image dimensions: {L}x{W_strip}")
print(f"Calculated Radius: {R:.2f}")
print(f"Calculated Column Height: {H:.2f}")

# Subdivisions
rows = 20 # Across the width of the strip
cols = 2000 # Along the length of the strip

# Use multisampling for smoother edges if available
config = Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
try:
    window = pyglet.window.Window(1280, 720, "Joshua Roll Column - Helical Wrap", resizable=True, config=config)
except:
    window = pyglet.window.Window(1280, 720, "Joshua Roll Column - Helical Wrap", resizable=True)

# Load texture using PIL
print("Loading and processing texture...")
with Image.open(texture_file) as img:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Flip to match OpenGL UV coordinates
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    width, height = img.size
    texture_data = img.tobytes()
    
texture_image = pyglet.image.ImageData(width, height, 'RGB', texture_data)
texture = texture_image.get_texture()

# Handle power-of-two padding in Pyglet 1.5
# texture might be a TextureRegion if it's NPOT
if hasattr(texture, 'owner'):
    # It's a TextureRegion
    owner = texture.owner
    # tex_coords is (u1, v1, r1, u2, v2, r2, u3, v3, r3, u4, v4, r4)
    # Typically u1=0, v1=0, u2=max_u, v2=0, u3=max_u, v3=max_v, u4=0, v4=max_v
    max_u = texture.tex_coords[3]
    max_v = texture.tex_coords[7]
    target = owner.target
    texture_id = owner.id
else:
    # It's a full Texture
    max_u = 1.0
    max_v = 1.0
    target = texture.target
    texture_id = texture.id

print(f"Texture UV Scale: {max_u:.4f}, {max_v:.4f}")

# GL State
glClearColor(1, 1, 1, 1)  # White background
glEnable(GL_TEXTURE_2D)
glBindTexture(target, texture_id)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glEnable(GL_DEPTH_TEST)

# Create the mesh
vertices = []
uvs = []
indices = []

for i in range(rows + 1):
    v_norm = i / rows
    t_prime = v_norm * W_strip 
    for j in range(cols + 1):
        u_norm = j / cols
        s = u_norm * L 
        
        # Helical mapping
        x_unrolled = s * math.cos(angle_rad) - t_prime * math.sin(angle_rad)
        z = s * math.sin(angle_rad) + t_prime * math.cos(angle_rad)
        
        phi = x_unrolled / R
        x = R * math.cos(phi)
        y = R * math.sin(phi)
        
        vertices.extend([x, y, z])
        uvs.extend([u_norm * max_u, v_norm * max_v])

for i in range(rows):
    for j in range(cols):
        idx0 = i * (cols + 1) + j
        idx1 = idx0 + 1
        idx2 = (i + 1) * (cols + 1) + j
        idx3 = idx2 + 1
        indices.extend([idx0, idx1, idx2, idx1, idx3, idx2])

batch = pyglet.graphics.Batch()
vlist = batch.add_indexed(len(vertices)//3, GL_TRIANGLES, None, indices,
    ('v3f', vertices),
    ('t2f', uvs)
)

def export_to_obj(obj_filename, vertices, uvs, indices, texture_filename):
    mtl_filename = obj_filename.replace('.obj', '.mtl')
    
    print(f"Exporting model to {obj_filename}...")
    
    # Write MTL file
    with open(mtl_filename, 'w') as f:
        f.write("newmtl ColumnMaterial\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write(f"map_Kd {texture_filename}\n")
    
    # Write OBJ file
    with open(obj_filename, 'w') as f:
        f.write(f"mtllib {mtl_filename}\n")
        f.write("o TriumphalColumn\n")
        
        # Vertices
        for i in range(0, len(vertices), 3):
            f.write(f"v {vertices[i]:.4f} {vertices[i+1]:.4f} {vertices[i+2]:.4f}\n")
            
        # UVs (Need to normalize these based on max_u/max_v for standard OBJ readers)
        # However, our 'uvs' list already contains the scaled values (u_norm * max_u).
        # Standard OBJ readers expect 0.0 to 1.0. 
        # Since we scaled them for OpenGL padded textures, we should probably 
        # export them unscaled (0 to 1) so they work in external software 
        # with the original image.
        for i in range(0, len(uvs), 2):
            # To export correctly for external tools using the raw image:
            u_export = uvs[i] / max_u if max_u != 0 else 0
            v_export = uvs[i+1] / max_v if max_v != 0 else 0
            f.write(f"vt {u_export:.6f} {v_export:.6f}\n")
            
        # Faces (1-indexed)
        f.write("usemtl ColumnMaterial\n")
        for i in range(0, len(indices), 3):
            # OBJ indices are 1-based
            v1, v2, v3 = indices[i]+1, indices[i+1]+1, indices[i+2]+1
            f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
            
    print(f"Export complete. MTL saved as {mtl_filename}")

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.E:
        export_to_obj("joshua_roll_column.obj", vertices, uvs, indices, texture_file)

# Viewing parameters
rot_x, rot_y = -90, 0
cam_dist = H * 1.2
center_z = H / 2
view_z_offset = 0

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window.width/window.height, 10, H * 10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glTranslatef(0, 0, -cam_dist)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 0, 1)
    
    # Apply vertical offset and center the column
    glTranslatef(0, 0, -(center_z + view_z_offset))
    
    batch.draw()

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global rot_x, rot_y, view_z_offset
    # Right button or Shift+Left to pan up/down
    if (buttons & pyglet.window.mouse.RIGHT) or (modifiers & pyglet.window.key.MOD_SHIFT):
        view_z_offset += dy * (H / 1000.0)
    else:
        rot_x -= dy * 0.2
        rot_y += dx * 0.2

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global cam_dist
    cam_dist *= 1.1 ** (-scroll_y)

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)

def update(dt):
    pass

pyglet.clock.schedule_interval(update, 1/60.0)
print("\nInstructions:")
print("- Mouse Drag: Rotate")
print("- Right-Click Drag (or Shift-Drag): Move Up/Down")
print("- Scroll: Zoom")
print("- Press 'E': Export to OBJ (joshua_roll_column.obj)")
pyglet.app.run()
