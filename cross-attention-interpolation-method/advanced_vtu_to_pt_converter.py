# --------------------------------------------------------------
# app.py – VTU → PT Converter (Relative Path, os.path.join)
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
st.markdown("Convert `.vtu` → `.pt` with **3D preview** – **local folder only**.")

# --------------------------------------------------------------
# 1. Define DATA_ROOT using os.path.join + __file__
# --------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "laser_simulations")

if not os.path.exists(DATA_ROOT):
    st.error(
        f"Folder not found: `{DATA_ROOT}`\n\n"
        "Make sure:\n"
        "1. `laser_simulations/` is in the **same folder** as `app.py`\n"
        "2. It contains subfolders like `P10_V65/`, `P35_V65/`, etc.\n"
        "3. Each subfolder has `.vtu` files"
    )
    st.stop()

# --------------------------------------------------------------
# 2. Detect simulations
# --------------------------------------------------------------
def find_simulations(root):
    pattern = re.compile(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", re.IGNORECASE)
    sims = []
    for entry in os.scandir(root):
        if entry.is_dir() and pattern.match(entry.name):
            vtu_files = glob.glob(os.path.join(entry.path, "*.vtu"))
            if vtu_files:
                P = float(pattern.match(entry.name).group(1))
                V = float(pattern.match(entry.name).group(2))
                sims.append({
                    "name": entry.name,
                    "path": entry.path,
                    "P": P,
                    "V": V,
                    "files": len(vtu_files),
                })
    return sorted(sims, key=lambda x: x["name"])

simulations = find_simulations(DATA_ROOT)

if not simulations:
    st.warning("No `Pxx_Vyy` folders with `.vtu` files found.")
    st.stop()

st.success(f"Found {len(simulations)} simulations:")
cols = st.columns(3)
for i, sim in enumerate(simulations):
    with cols[i % 3]:
        st.markdown(
            f"**{sim['name']}**  \nP = `{sim['P']}` W  \nV = `{sim['V']}` mm/s  \nFiles: `{sim['files']}`"
        )

# --------------------------------------------------------------
# 3. Select & 3D Preview
# --------------------------------------------------------------
selected_names = st.multiselect(
    "Select simulations to convert",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]]
)

if selected_names:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_sample = sorted(glob.glob(os.path.join(first_sim["path"], "*.vtu")))[0]
    st.write(f"**3D Preview**: `{os.path.basename(vtu_sample)}`")

    @st.cache_data
    def load_preview_mesh(p):
        return pv.read(p)

    mesh = load_preview_mesh(vtu_sample)

    # Find temperature field
    temp_field = None
    for key in mesh.point_data.keys():
        if "temp" in key.lower() or key.lower() in ("t", "temperature"):
            temp_field = key
            break
    if temp_field is None and mesh.point_data:
        temp_field = list(mesh.point_data.keys())[0]

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, scalars=temp_field, cmap="hot", show_scalar_bar=True)
    plotter.set_background("white")
    plotter.camera_position = 'xy'
    png = plotter.screenshot(transparent_background=True, return_img=True)
    st.image(png, use_column_width=True)

# --------------------------------------------------------------
# 4. Optional: Split to ≤25 MiB
# --------------------------------------------------------------
split_parts = st.checkbox("Split .pt into ≤25 MiB parts (GitHub-safe)", value=True)
max_mb = st.slider("Max size per part (MiB)", 5, 25, 20) if split_parts else 25
MAX_BYTES = max_mb * 1024 * 1024

# --------------------------------------------------------------
# 5. Convert Button
# --------------------------------------------------------------
if st.button("Convert to .pt", type="primary"):
    OUTPUT_ROOT = Path(os.path.join(SCRIPT_DIR, "processed_pt"))
    OUTPUT_ROOT.mkdir(exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_download_files = []

    for idx, name in enumerate(selected_names):
        sim = next(s for s in simulations if s["name"] == name)
        status_text.text(f"Processing `{name}`...")

        # --- Load all .vtu ---
        vtu_files = sorted(glob.glob(os.path.join(sim["path"], "*.vtu")))
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
        tensors = {col: torch.from_numpy(full_df[col].values.astype(np.float32)) for col in numeric_cols}
        metadata = {"simulation": name, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}

        # --- Split into parts ---
        N = next(iter(tensors.values())).shape[0]
        row_bytes = sum(t[0:1].numel() * 4 for t in tensors.values())
        rows_per_part = max(1, MAX_BYTES // row_bytes)
        n_parts = (N + rows_per_part - 1) // rows_per_part

        sim_out_dir = OUTPUT_ROOT / name
        sim_out_dir.mkdir(exist_ok=True)

        for i in range(n_parts):
            start = i * rows_per_part
            end = min(start + rows_per_part, N)

            part_tensors = {k: v[start:end] for k, v in tensors.items()}
            part_data = {**part_tensors, **metadata,
                         "part_index": i, "total_parts": n_parts,
                         "row_start": start, "row_end": end}

            part_file = sim_out_dir / f"part_{i:04d}.pt"
            torch.save(part_data, part_file)
            all_download_files.append(part_file)

        progress_bar.progress((idx + 1) / len(selected_names))

    status_text.success(f"Converted & split {len(selected_names)} simulation(s)!")
    st.balloons()

    # --------------------------------------------------------------
    # 6. Download All Parts
    # --------------------------------------------------------------
    st.subheader("Download Split .pt Files (≤25 MiB each)")
    total_parts = len(all_download_files)
    total_size_gb = sum(f.stat().st_size for f in all_download_files) / (1024**3)
    st.write(f"**{total_parts} parts** | **{total_size_gb:.2f} GB total**")

    for part_path in all_download_files:
        size_mb = part_path.stat().st_size / (1024**2)
        rel_path = part_path.relative_to(OUTPUT_ROOT)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"`{rel_path}` — **{size_mb:.2f} MiB**")
        with col2:
            with open(part_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name=part_path.name,
                    mime="application/octet-stream",
                    key=str(part_path)
                )

    st.info(
        f"All files saved in: `{OUTPUT_ROOT}`\n\n"
        "Upload the **entire folder** to GitHub — **no LFS needed**!"
    )

