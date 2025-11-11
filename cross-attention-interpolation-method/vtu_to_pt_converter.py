# --------------------------------------------------------------
# app.py – GUI .vtu → .pt Converter (Streamlit + 3D Preview)
# --------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import torch
import pyvista as pv
import panel as pn
import glob
import re
import os
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT Converter", layout="wide")
st.title("VTU to PyTorch (.pt) Converter")
st.markdown("Convert laser heating `.vtu` files → ML-ready `.pt` tensors with **3D preview**.")

# --------------------------------------------------------------
# 1. Folder Input
# --------------------------------------------------------------
data_folder = st.text_input(
    "Folder with Pxx_Vyy subfolders",
    value="laser_simulations",
    help="e.g. laser_simulations/P10_V65/*.vtu"
)
DATA_ROOT = Path(data_folder)

if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`")
    st.stop()

# --------------------------------------------------------------
# 2. Detect simulations
# --------------------------------------------------------------
def find_simulations(root):
    pattern = re.compile(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", re.IGNORECASE)
    sims = []
    for p in root.iterdir():
        if p.is_dir() and pattern.match(p.name):
            vtu_files = list(p.glob("*.vtu"))
            if vtu_files:
                P = float(pattern.match(p.name).group(1))
                V = float(pattern.match(p.name).group(2))
                sims.append({"name": p.name, "path": p, "P": P, "V": V, "files": len(vtu_files)})
    return sorted(sims, key=lambda x: x["name"])

simulations = find_simulations(DATA_ROOT)

if not simulations:
    st.warning("No `Pxx_Vyy` folders with `.vtu` files found.")
    st.stop()

st.success(f"Found {len(simulations)} simulations:")
cols = st.columns(3)
for i, sim in enumerate(simulations):
    with cols[i % 3]:
        st.markdown(f"**{sim['name']}**  \nP = `{sim['P']}` W  \nV = `{sim['V']}` mm/s  \nFiles: `{sim['files']}`")

# --------------------------------------------------------------
# 3. Select & Preview
# --------------------------------------------------------------
selected_names = st.multiselect(
    "Select simulations to convert",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]]
)

# Preview first .vtu of first selected
if selected_names:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_sample = sorted(first_sim["path"].glob("*.vtu"))[0]
    st.write(f"**3D Preview**: `{vtu_sample.name}`")

    @st.cache_data
    def load_preview_mesh(path):
        mesh = pv.read(path)
        return mesh

    mesh = load_preview_mesh(vtu_sample)

    # Extract temperature if available
    temp_field = None
    for key in mesh.point_data.keys():
        if "temp" in key.lower() or "t" == key.lower():
            temp_field = key
            break
    if temp_field is None and mesh.point_data:
        temp_field = list(mesh.point_data.keys())[0]

    # Plot with Panel + PyVista
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, scalars=temp_field, cmap="hot", show_scalar_bar=True)
    plotter.set_background("white")
    plotter.camera_position = 'xy'
    png = plotter.screenshot(transparent_background=True, return_img=True)

    st.image(png, use_column_width=True)

# --------------------------------------------------------------
# 4. Convert Button
# --------------------------------------------------------------
if st.button("Convert to .pt", type="primary"):
    OUTPUT_DIR = Path("processed_pt")
    OUTPUT_DIR.mkdir(exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    for idx, name in enumerate(selected_names):
        sim = next(s for s in simulations if s["name"] == name)
        status_text.text(f"Processing {name}...")

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
        tensor_dict = {
            col: torch.from_numpy(full_df[col].values.astype(np.float32))
            for col in numeric_cols
        }
        metadata = {
            "simulation": name,
            "P_W": float(sim["P"]),
            "Vscan_mm_s": float(sim["V"]),
        }
        save_data = {**tensor_dict, **metadata}

        pt_path = OUTPUT_DIR / f"{name}.pt"
        torch.save(save_data, pt_path)
        results.append(pt_path)

        progress_bar.progress((idx + 1) / len(selected_names))

    status_text.success(f"Converted {len(results)} simulations!")
    st.balloons()

    # --------------------------------------------------------------
    # 5. Download Results
    # --------------------------------------------------------------
    st.subheader("Download .pt Files")
    for pt_file in results:
        with open(pt_file, "rb") as f:
            st.download_button(
                label=f"{pt_file.name} ({pt_file.stat().st_size / 1e6:.1f} MB)",
                data=f,
                file_name=pt_file.name,
                mime="application/octet-stream"
            )

    st.info(f"All `.pt` files saved in: `{OUTPUT_DIR}`")
