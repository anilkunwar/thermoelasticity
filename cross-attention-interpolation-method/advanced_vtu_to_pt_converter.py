# --------------------------------------------------------------
# advanced_vtu_to_pt_converter.py – with 3D rendering support
# --------------------------------------------------------------
import os
import io
import re
import glob
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# 1. ENVIRONMENT CONFIGURATION (headless-safe but 3D-enabled)
# ==============================================================

# Enable PyVista off-screen rendering (OSMesa fallback)
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_USE_PANEL"] = "True"
os.environ["PYVISTA_AUTO_CLOSE"] = "False"
os.environ["PYVISTA_BUILD_TYPE"] = "headless"

# Use OSMesa software rendering if OpenGL unavailable
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["VTK_USE_OFFSCREEN"] = "True"

# Disable memory leak detection in VTK (saves memory on cloud)
os.environ["PYVISTA_DISABLE_FOONATHAN_MEMORY"] = "1"

import pyvista as pv

# Start virtual framebuffer (for cloud/headless)
try:
    pv.start_xvfb()
    print("[INFO] Virtual framebuffer started for headless rendering.")
except Exception as e:
    pv.OFF_SCREEN = True
    print(f"[INFO] Xvfb not available, using OSMesa software rendering: {e}")

# ==============================================================
# 2. STREAMLIT APP HEADER
# ==============================================================

st.set_page_config(page_title="3D VTU → PT Converter", layout="wide")
st.title("3D VTU → PyTorch Converter")
st.markdown("Convert `.vtu` simulation files to `.pt` format and preview 3D meshes interactively.")

# --------------------------------------------------------------
# 3. DETECT DATA DIRECTORY
# --------------------------------------------------------------

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = "laser_simulations"

data_folder = st.text_input(
    "Folder containing your Pxx_Vyy subfolders:",
    value=DEFAULT_FOLDER,
    help="Each folder should contain `.vtu` files like `a_t0001.vtu`."
)

DATA_ROOT = SCRIPT_DIR / data_folder
if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`")
    st.stop()

# --------------------------------------------------------------
# 4. DETECT SIMULATION FOLDERS
# --------------------------------------------------------------

def find_simulations(root: Path):
    pattern = re.compile(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", re.IGNORECASE)
    sims = []
    for p in root.iterdir():
        if p.is_dir() and pattern.match(p.name):
            vtu_files = list(p.glob("*.vtu"))
            if vtu_files:
                P = float(pattern.match(p.name).group(1))
                V = float(pattern.match(p.name).group(2))
                sims.append({
                    "name": p.name,
                    "path": p,
                    "P": P,
                    "V": V,
                    "files": len(vtu_files),
                })
    return sorted(sims, key=lambda x: x["name"])

simulations = find_simulations(DATA_ROOT)
if not simulations:
    st.warning("No simulation folders found.")
    st.stop()

st.success(f"Found {len(simulations)} simulations.")
cols = st.columns(3)
for i, sim in enumerate(simulations):
    with cols[i % 3]:
        st.markdown(f"**{sim['name']}**  \nP={sim['P']}W  \nV={sim['V']}mm/s  \nFiles={sim['files']}")

# --------------------------------------------------------------
# 5. SELECT SIMULATION AND LOAD SAMPLE MESH
# --------------------------------------------------------------

selected_names = st.multiselect(
    "Select simulations to preview/convert:",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]],
)

if not selected_names:
    st.stop()

first_sim = next(s for s in simulations if s["name"] == selected_names[0])
vtu_sample = sorted(first_sim["path"].glob("*.vtu"))[0]
st.write(f"**3D Preview:** `{vtu_sample.name}`")

# --------------------------------------------------------------
# 6. LOAD AND VISUALIZE 3D MESH (FULL RENDERING)
# --------------------------------------------------------------

@st.cache_data
def load_mesh(path):
    import meshio
    try:
        mesh = pv.read(path)
        return mesh
    except Exception:
        m = meshio.read(str(path))
        class SimpleMesh:
            pass
        sm = SimpleMesh()
        sm.points = m.points
        sm.point_data = getattr(m, "point_data", {})
        sm.field_data = getattr(m, "field_data", {})
        return sm

mesh = load_mesh(vtu_sample)
if mesh is None:
    st.error("Failed to load mesh.")
    st.stop()

# Pick temperature-like field
temp_field = None
try:
    keys = list(mesh.point_data.keys())
except Exception:
    keys = []
for k in keys:
    if "temp" in k.lower() or k.lower() in ("t", "temperature"):
        temp_field = k
        break
if temp_field is None and keys:
    temp_field = keys[0]

st.write(f"Using scalar field: `{temp_field}`")

# Render 3D preview (interactive)
try:
    plotter = pv.Plotter(off_screen=True, window_size=[900, 700])
    plotter.add_mesh(mesh, scalars=temp_field, cmap="inferno", show_edges=False)
    plotter.set_background("white")
    plotter.show_bounds(grid=True)
    plotter.camera_position = "xy"
    png = plotter.screenshot(return_img=True)
    plotter.close()
    st.image(png, caption="3D Mesh Preview (rendered with OSMesa)", use_container_width=True)
except Exception as e:
    st.warning(f"3D rendering failed: {e}")
    st.stop()

# --------------------------------------------------------------
# 7. CONVERT SELECTED SIMULATIONS TO .PT
# --------------------------------------------------------------

split_parts = st.checkbox("Split large files into parts (<25MB)", value=True)
max_mb = st.slider("Max part size (MiB)", 5, 25, 20) if split_parts else 200
MAX_BYTES = max_mb * 1024 * 1024

if st.button("Convert to .pt", type="primary"):
    OUTPUT_ROOT = SCRIPT_DIR / "processed_pt"
    OUTPUT_ROOT.mkdir(exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_pt_files = []

    for idx, name in enumerate(selected_names):
        sim = next(s for s in simulations if s["name"] == name)
        status_text.text(f"Processing `{name}`…")

        vtu_files = sorted(sim["path"].glob("*.vtu"))
        frames = []

        for vtu_path in tqdm(vtu_files, desc=name, leave=False):
            mesh = pv.read(vtu_path)
            time_val = mesh.field_data.get("TimeValue", [np.nan])[0]
            points = mesh.points
            base_data = {
                "simulation": name,
                "P_W": sim["P"],
                "Vscan_mm_s": sim["V"],
                "time": time_val,
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
            }
            for field_name in mesh.point_data:
                arr = mesh.point_data[field_name]
                if arr.ndim == 1:
                    base_data[field_name] = arr
                else:
                    for i in range(arr.shape[1]):
                        base_data[f"{field_name}_{i}"] = arr[:, i]
            df = pd.DataFrame(base_data)
            frames.append(df)

        full_df = pd.concat(frames, ignore_index=True)
        numeric_cols = full_df.select_dtypes(include=[np.number]).columns

        tensors = {
            col: torch.from_numpy(full_df[col].values.astype(np.float32))
            for col in numeric_cols
        }
        metadata = {"simulation": name, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}

        # Split into chunks
        N = next(iter(tensors.values())).shape[0]
        row_bytes = sum(t[0:1].numel() * 4 for t in tensors.values())
        rows_per_part = max(1, MAX_BYTES // row_bytes)
        n_parts = (N + rows_per_part - 1) // rows_per_part

        sim_dir = OUTPUT_ROOT / name
        sim_dir.mkdir(exist_ok=True)

        for i in range(n_parts):
            start = i * rows_per_part
            end = min(start + rows_per_part, N)
            part = {k: v[start:end] for k, v in tensors.items()}
            part.update(metadata)
            part["part_index"] = i
            part["total_parts"] = n_parts
            part_file = sim_dir / f"part_{i:04d}.pt"
            torch.save(part, part_file)
            all_pt_files.append(part_file)

        progress_bar.progress((idx + 1) / len(selected_names))

    status_text.success(f"Converted {len(selected_names)} simulations!")
    st.balloons()

    st.subheader("Download .pt Files")
    for pt_file in all_pt_files:
        size_mb = pt_file.stat().st_size / (1024 * 1024)
        with open(pt_file, "rb") as f:
            st.download_button(
                label=f"{pt_file.relative_to(OUTPUT_ROOT)} ({size_mb:.1f} MB)",
                data=f,
                file_name=pt_file.name,
                mime="application/octet-stream",
            )
    st.info(f"Files saved in `{OUTPUT_ROOT}`")

