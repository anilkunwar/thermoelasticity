# --------------------------------------------------------------
# app.py – VTU → PT Converter (stpyvista + Streamlit Cloud)
# --------------------------------------------------------------
import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# 1. PURE OSMESA HEADLESS RENDERING – NO X11
# ==============================================================
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_AUTO_CLOSE"] = "False"
os.environ["PYVISTA_USE_PANEL"] = "True"
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
st.markdown(
    """
    **Interactive 3D preview** with `stpyvista` | **≤25 MiB split** | **GitHub-ready**
    """
)

# --------------------------------------------------------------
# 2. DATA_ROOT – relative to script
# --------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = "laser_simulations"

data_folder = st.text_input(
    "Folder with Pxx_Vyy sub-folders",
    value=DEFAULT_FOLDER,
    help="Leave default if next to `app.py`."
)
DATA_ROOT = SCRIPT_DIR / data_folder

if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`\n\nUpload **laser_simulations/** next to `app.py`.")
    st.stop()

# --------------------------------------------------------------
# 3. Detect simulations
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
    st.warning("No `Pxx_Vyy` folders found.")
    st.stop()

st.success(f"Found {len(simulations)} simulations:")
cols = st.columns(3)
for i, sim in enumerate(simulations):
    with cols[i % 3]:
        st.markdown(f"**{sim['name']}**  \nP = `{sim['P']}` W  \nV = `{sim['V']}` mm/s  \nFiles: `{sim['files']}`")

# --------------------------------------------------------------
# 4. Interactive 3D Preview with stpyvista
# --------------------------------------------------------------
try:
    from stpyvista import stpyvista
    HAS_STPYVISTA = True
except ImportError:
    HAS_STPYVISTA = False
    st.warning("`stpyvista` not available. Install with `pip install stpyvista`.")

selected_names = st.multiselect(
    "Select simulations",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]],
)

if selected_names and HAS_STPYVISTA:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_sample = sorted(first_sim["path"].glob("*.vtu"))[0]
    st.write(f"**Interactive 3D**: `{vtu_sample.name}`")

    @st.cache_resource
    def load_mesh(path):
        mesh = pv.read(path)
        # Pick temperature field
        temp_field = None
        for k in mesh.point_data:
            if "temp" in k.lower() or k.lower() in ("t", "temperature"):
                temp_field = k
                break
        if temp_field is None and mesh.point_data:
            temp_field = list(mesh.point_data.keys())[0]
        mesh.set_active_scalars(temp_field)
        return mesh

    mesh = load_mesh(vtu_sample)

    stpyvista(
        mesh,
        panel_kwargs=dict(
            orientation_widget=True,
            background="white",
            zoom=1.6,
            style="surface",
        ),
        use_container_width=True,
        key="stpyvista_preview"
    )
else:
    st.info("Select a simulation to see 3D preview.")

# --------------------------------------------------------------
# 5. Convert + Split
# --------------------------------------------------------------
split_parts = st.checkbox("Split into ≤25 MiB parts", value=True)
max_mb = st.slider("Max part size (MiB)", 5, 25, 20) if split_parts else 200
MAX_BYTES = max_mb * 1024 * 1024

if st.button("Convert to .pt", type="primary"):
    OUTPUT_ROOT = SCRIPT_DIR / "processed_pt"
    OUTPUT_ROOT.mkdir(exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_files = []

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
        tensors = {col: torch.from_numpy(full_df[col].values.astype(np.float32)) for col in numeric_cols}
        metadata = {"simulation": name, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}

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
            all_files.append(part_file)

        progress_bar.progress((idx + 1) / len(selected_names))

    status_text.success(f"Converted {len(selected_names)} simulation(s)!")
    st.balloons()

    # --------------------------------------------------------------
    # 6. Download
    # --------------------------------------------------------------
    st.subheader("Download .pt Parts")
    for f in all_files:
        size_mb = f.stat().st_size / (1024**2)
        with open(f, "rb") as fp:
            st.download_button(
                label=f"{f.relative_to(OUTPUT_ROOT)} ({size_mb:.1f} MB)",
                data=fp,
                file_name=f.name,
                mime="application/octet-stream"
            )
    st.info(f"Saved in: `{OUTPUT_ROOT}`")

# --------------------------------------------------------------
# 7. Reconstruct
# --------------------------------------------------------------
with st.expander("Reconstruct full .pt"):
    st.code(
        """
import torch, glob, os
def load_simulation(folder):
    parts = sorted(glob.glob(os.path.join(folder, "part_*.pt")))
    tensors = {}
    meta = None
    for p in parts:
        d = torch.load(p)
        if meta is None:
            meta = {k:v for k,v in d.items() if not isinstance(v,torch.Tensor)}
        for k,v in d.items():
            if isinstance(v,torch.Tensor):
                tensors.setdefault(k,[]).append(v)
    full = {k:torch.cat(v) for k,v in tensors.items()}
    full.update(meta)
    return full
        """,
        language="python"
    )
