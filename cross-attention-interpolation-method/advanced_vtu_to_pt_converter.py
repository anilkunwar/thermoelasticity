import os
import pyvista as pv
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Headless rendering configuration (Streamlit Cloud safe)
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_USE_PANEL"] = "True"
os.environ["PYVISTA_AUTO_CLOSE"] = "False"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "True"
os.environ["VTK_USE_OFFSCREEN"] = "True"
os.environ["PYVISTA_BUILD_TYPE"] = "headless"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["PYVISTA_DISABLE_FOONATHAN_MEMORY"] = "1"

try:
    pv.start_xvfb()
    print("[INFO] Xvfb virtual display started.")
except Exception as e:
    pv.OFF_SCREEN = True
    print(f"[INFO] Xvfb not available. Using off-screen mode. ({e})")

st.set_page_config(page_title="VTU 3D Viewer", layout="wide")
st.title("VTU 3D Viewer with PyVista")

# VTU file selector
DATA_FOLDER = Path("./laser_simulations")  # Change this to your folder with VTU files
vtu_files = list(DATA_FOLDER.glob("**/*.vtu"))

if not vtu_files:
    st.error(f"No .vtu files found in {DATA_FOLDER}")
    st.stop()

selected_file = st.selectbox("Select a VTU file", options=[str(f) for f in vtu_files])

# Load the mesh
@st.cache_data
def load_mesh(path):
    try:
        return pv.read(path)
    except Exception as e:
        st.error(f"Failed to load VTU mesh: {e}")
        return None

mesh = load_mesh(selected_file)

if mesh is None:
    st.stop()

# Find suitable scalar field for coloring
def find_scalar(mesh):
    try:
        keys = list(mesh.point_data.keys())
    except Exception:
        keys = []
    for key in keys:
        if "temp" in key.lower() or key.lower() in ("t", "temperature"):
            return key
    if keys:
        return keys[0]
    return None

scalar_field = find_scalar(mesh)

# Render 3D mesh off-screen and show screenshot
try:
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(mesh, scalars=scalar_field, cmap="hot", show_scalar_bar=True)
    plotter.set_background("white")
    plotter.camera_position = "xy"
    img = plotter.screenshot(transparent_background=True, return_img=True)
    plotter.close()
    st.image(img, use_column_width=True)
except Exception as e:
    st.warning("3D preview unavailable, showing 2D projection fallback.")
    pts = np.asarray(mesh.points)
    if scalar_field and scalar_field in mesh.point_data:
        vals = np.asarray(mesh.point_data[scalar_field])
        if vals.ndim > 1:
            vals = np.linalg.norm(vals, axis=1)
    else:
        vals = pts[:, 2]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=6, cmap="hot", marker=".", rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{Path(selected_file).name} — 2D Projection (XY)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, label=(scalar_field or "value"))
    plt.tight_layout()
    st.pyplot(fig)
