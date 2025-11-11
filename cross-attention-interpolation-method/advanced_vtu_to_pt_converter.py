# --------------------------------------------------------------
# app.py – VTU → PT (stpyvista + auto-refresh + robust)
# --------------------------------------------------------------
import os
import re
import torch
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# 1. HEADLESS OSMESA (no X11)
# ==============================================================
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_AUTO_CLOSE"] = "False"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["PYVISTA_HEADLESS"] = "True"

import pyvista as pv
pv.global_theme.background = "white"

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT", layout="wide")
st.title("VTU → PyTorch (.pt) Converter")
st.markdown("**Interactive 3D** | **≤25 MiB split** | **Auto-refresh**")

# --------------------------------------------------------------
# 2. DATA_ROOT
# --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "laser_simulations"

if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`\n\nUpload **laser_simulations/** next to `app.py`.")
    st.stop()

# --------------------------------------------------------------
# 3. Find simulations (cached)
# --------------------------------------------------------------
@st.cache_data
def find_simulations():
    pattern = re.compile(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", re.IGNORECASE)
    sims = []
    for p in DATA_ROOT.iterdir():
        if p.is_dir() and pattern.match(p.name):
            vtu_files = list(p.glob("*.vtu"))
            if vtu_files:
                P = float(pattern.match(p.name).group(1))
                V = float(pattern.match(p.name).group(2))
                sims.append({"name": p.name, "path": p, "P": P, "V": V, "files": len(vtu_files)})
    return sorted(sims, key=lambda x: x["name"])

simulations = find_simulations()
if not simulations:
    st.warning("No simulations found.")
    st.stop()

st.success(f"Found {len(simulations)} simulations")

# --------------------------------------------------------------
# 4. Select + Interactive 3D (stpyvista + key + cache)
# --------------------------------------------------------------
selected = st.selectbox(
    "Select simulation",
    options=[s["name"] for s in simulations],
    format_func=lambda x: f"{x} | P={next(s for s in simulations if s['name']==x)['P']}W, V={next(s for s in simulations if s['name']==x)['V']}mm/s",
    key="sim_select"
)

sim = next(s for s in simulations if s["name"] == selected)
vtu_sample = sorted(sim["path"].glob("*.vtu"))[0]

st.write(f"**3D Preview**: `{vtu_sample.name}`")

try:
    from stpyvista import stpyvista
    HAS_STPY = True
except ImportError:
    HAS_STPY = False
    st.warning("`stpyvista` not installed.")

@st.cache_resource
def load_mesh(_path):
    mesh = pv.read(_path)
    temp_field = next((k for k in mesh.point_data if "temp" in k.lower()), None)
    if temp_field is None and mesh.point_data:
        temp_field = list(mesh.point_data.keys())[0]
    mesh.set_active_scalars(temp_field)
    # Downsample if too big
    if mesh.n_points > 200_000:
        mesh = mesh.decimate(0.7)
    return mesh

mesh = load_mesh(vtu_sample)

if HAS_STPY:
    with st.spinner("Rendering 3D..."):
        stpyvista(
            mesh,
            panel_kwargs=dict(orientation_widget=True, background="white", zoom=1.5),
            use_container_width=True,
            key=f"preview_{selected}"
        )
else:
    st.info("Install `stpyvista` for interactive 3D.")

# --------------------------------------------------------------
# 5. Convert Button + Progress
# --------------------------------------------------------------
if st.button("Convert to .pt", type="primary", key="convert_btn"):
    OUTPUT_ROOT = SCRIPT_DIR / "processed_pt"
    OUTPUT_ROOT.mkdir(exist_ok=True)
    sim_dir = OUTPUT_ROOT / selected
    sim_dir.mkdir(exist_ok=True)

    vtu_files = sorted(sim["path"].glob("*.vtu"))
    progress = st.progress(0)
    status = st.empty()
    frames = []

    for i, vtu_path in enumerate(vtu_files):
        status.text(f"Reading {vtu_path.name}...")
        mesh = pv.read(vtu_path)
        time_val = mesh.field_data.get("TimeValue", [np.nan])[0]
        points = mesh.points

        base_data = {
            "simulation": selected,
            "P_W": sim["P"],
            "Vscan_mm_s": sim["V"],
            "time": time_val,
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
        }
        for k, arr in mesh.point_data.items():
            if arr.ndim == 1:
                base_data[k] = arr
            else:
                for j in range(arr.shape[1]):
                    base_data[f"{k}_{j}"] = arr[:, j]
        frames.append(pd.DataFrame(base_data))
        progress.progress((i + 1) / len(vtu_files))

    status.text("Concatenating...")
    full_df = pd.concat(frames, ignore_index=True)
    tensors = {c: torch.from_numpy(full_df[c].values.astype(np.float32)) 
               for c in full_df.select_dtypes(include=[np.number]).columns}
    metadata = {"simulation": selected, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}

    # Split into ≤25 MiB
    MAX_BYTES = 25 * 1024 * 1024
    N = next(iter(tensors.values())).shape[0]
    row_bytes = sum(t[0:1].numel() * 4 for t in tensors.values())
    rows_per_part = max(1, MAX_BYTES // row_bytes)
    n_parts = (N + rows_per_part - 1) // rows_per_part

    part_files = []
    for i in range(n_parts):
        start = i * rows_per_part
        end = min(start + rows_per_part, N)
        part = {k: v[start:end] for k, v in tensors.items()}
        part.update(metadata)
        part["part_index"] = i
        part["total_parts"] = n_parts
        pt_path = sim_dir / f"part_{i:04d}.pt"
        torch.save(part, pt_path)
        part_files.append(pt_path)

    progress.empty()
    status.success(f"Done! {len(part_files)} parts saved.")
    st.balloons()

    # --------------------------------------------------------------
    # 6. Download
    # --------------------------------------------------------------
    st.subheader("Download .pt Parts")
    for f in part_files:
        with open(f, "rb") as fp:
            st.download_button(
                label=f"{f.name} ({f.stat().st_size / 1e6:.1f} MB)",
                data=fp,
                file_name=f.name,
                mime="application/octet-stream",
                key=f"dl_{f.name}"
            )
