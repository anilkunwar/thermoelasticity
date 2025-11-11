# --------------------------------------------------------------
# app.py – VTU → PT Converter (GitHub Cloud, os.path.join)
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
import zipfile
import io
import requests
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT Converter (GitHub)", layout="wide")
st.title("VTU to PyTorch (.pt) Converter – GitHub Cloud")
st.markdown(
    """
    The app **downloads** the `laser_simulations/` folder from a GitHub repository,
    converts the `.vtu` files to `.pt` tensors and lets you download the results.
    """
)

# --------------------------------------------------------------
# 1. GitHub repo input
# --------------------------------------------------------------
repo_input = st.text_input(
    "GitHub repository (owner/repo)",
    value="your-username/your-repo",   # <-- change to your repo
    help="e.g. curiosity/laser-heating"
)
branch = st.text_input("Branch / Tag", value="main")
subfolder = st.text_input(
    "Folder inside repo (optional)",
    value="laser_simulations",
    help="Leave empty if the folder is at the repo root"
)

# Build raw GitHub URL for the folder (zip download)
raw_zip_url = f"https://github.com/{repo_input}/archive/{branch}.zip"
st.caption(f"Will download: `{raw_zip_url}`")

# --------------------------------------------------------------
# 2. Download & extract the folder
# --------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading repository…")
def download_and_extract():
    resp = requests.get(raw_zip_url, stream=True)
    resp.raise_for_status()
    zip_bytes = io.BytesIO(resp.content)

    with zipfile.ZipFile(zip_bytes) as z:
        # repo root inside zip is <repo>-<branch>/
        repo_root = z.namelist()[0].split("/")[0]
        # extract only the wanted sub-folder
        target_prefix = os.path.join(repo_root, subfolder).replace("\\", "/")
        files = [n for n in z.namelist() if n.startswith(target_prefix)]
        extracted = {}
        for name in files:
            rel = os.path.relpath(name, target_prefix)
            if rel == ".":
                continue
            data = z.read(name)
            extracted[rel] = data
    return extracted, target_prefix

try:
    extracted_files, _ = download_and_extract()
except Exception as e:
    st.error(f"Failed to download repo: {e}")
    st.stop()

# Write extracted files to a temporary folder (Streamlit cache)
TMP_ROOT = Path("/tmp/github_vtu")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = TMP_ROOT / "laser_simulations"
DATA_ROOT.mkdir(exist_ok=True)

for rel_path, content in extracted_files.items():
    full_path = DATA_ROOT / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(content)

st.success(f"Downloaded & extracted `{subfolder}` → `{DATA_ROOT}`")

# --------------------------------------------------------------
# 3. Detect simulations (same logic, now uses os.path.join)
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
                sims.append(
                    {
                        "name": entry.name,
                        "path": Path(entry.path),
                        "P": P,
                        "V": V,
                        "files": len(vtu_files),
                    }
                )
    return sorted(sims, key=lambda x: x["name"])

simulations = find_simulations(str(DATA_ROOT))

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
# 4. Select & 3-D preview
# --------------------------------------------------------------
selected_names = st.multiselect(
    "Select simulations to convert",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]],
)

if selected_names:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_sample = sorted(first_sim["path"].glob("*.vtu"))[0]
    st.write(f"**3-D Preview**: `{vtu_sample.name}`")

    @st.cache_data
    def load_preview_mesh(p):
        return pv.read(p)

    mesh = load_preview_mesh(vtu_sample)

    # pick a temperature-like field
    temp_field = None
    for k in mesh.point_data.keys():
        if "temp" in k.lower() or k.lower() in ("t", "temperature"):
            temp_field = k
            break
    if temp_field is None and mesh.point_data:
        temp_field = list(mesh.point_data.keys())[0]

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, scalars=temp_field, cmap="hot", show_scalar_bar=True)
    plotter.set_background("white")
    plotter.camera_position = "xy"
    png = plotter.screenshot(transparent_background=True, return_img=True)
    st.image(png, use_column_width=True)

# --------------------------------------------------------------
# 5. Convert button
# --------------------------------------------------------------
if st.button("Convert to .pt", type="primary"):
    OUTPUT_DIR = Path("processed_pt")
    OUTPUT_DIR.mkdir(exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

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

    status_text.success(f"Converted {len(results)} simulation(s)!")
    st.balloons()

    # --------------------------------------------------------------
    # 6. Download results
    # --------------------------------------------------------------
    st.subheader("Download .pt Files")
    for pt_file in results:
        with open(pt_file, "rb") as f:
            st.download_button(
                label=f"{pt_file.name} ({pt_file.stat().st_size/1e6:.1f} MB)",
                data=f,
                file_name=pt_file.name,
                mime="application/octet-stream",
            )
    st.info(f"All `.pt` files saved in: `{OUTPUT_DIR}`")
