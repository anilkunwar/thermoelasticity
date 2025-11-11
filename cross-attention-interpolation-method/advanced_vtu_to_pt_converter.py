# --------------------------------------------------------------
# app.py – VTU → PT Converter (Plotly 3D + Reset Button)
# --------------------------------------------------------------
import os
import re
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT Converter", layout="wide")
st.title("VTU → PyTorch (.pt) Converter")
st.markdown(
    "Convert laser-heating `.vtu` files → ML-ready `.pt` tensors **with interactive 3-D Plotly preview**."
)

# --------------------------------------------------------------
# RESET BUTTON (Top-right)
# --------------------------------------------------------------
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Reset", help="Clear cache & reset GUI"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.success("Cache & GUI reset!")
        st.experimental_rerun()

# --------------------------------------------------------------
# 2. DATA_ROOT
# --------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = "laser_simulations"

data_folder = st.text_input(
    "Folder with Pxx_Vyy sub-folders",
    value=DEFAULT_FOLDER,
    help="Leave default if the folder is next to `app.py`.",
    key="data_folder_input"
)
DATA_ROOT = SCRIPT_DIR / data_folder

if not DATA_ROOT.exists():
    st.error(
        f"Folder not found: `{DATA_ROOT}`\n\n"
        "Make sure **laser_simulations** (with P10_V65, …) is next to `app.py`."
    )
    st.stop()

# --------------------------------------------------------------
# 3. Detect simulations
# --------------------------------------------------------------
@st.cache_data
def find_simulations(_root):
    pattern = re.compile(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", re.IGNORECASE)
    sims = []
    for p in _root.iterdir():
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
# 4. Select & Plotly 3-D preview
# --------------------------------------------------------------
selected_names = st.multiselect(
    "Select simulations to convert",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]],
    key="selected_sims"
)

if selected_names:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_sample = sorted(first_sim["path"].glob("*.vtu"))[0]
    st.write(f"**3-D Preview**: `{vtu_sample.name}`")

    @st.cache_data
    def load_vtu_for_preview(_p):
        import meshio
        return meshio.read(_p)

    mesh = load_vtu_for_preview(vtu_sample)

    # Choose scalar field
    temp_field = None
    for k in mesh.point_data.keys():
        if "temp" in k.lower() or k.lower() in ("t", "temperature"):
            temp_field = k
            break
    if temp_field is None and mesh.point_data:
        temp_field = list(mesh.point_data.keys())[0]

    # Extract points & values
    points = np.asarray(mesh.points)
    if temp_field and temp_field in mesh.point_data:
        raw = np.asarray(mesh.point_data[temp_field])
        values = np.linalg.norm(raw, axis=1) if raw.ndim > 1 else raw
    else:
        values = points[:, 2]

    # Down-sample if huge
    MAX_POINTS = 200_000
    if points.shape[0] > MAX_POINTS:
        idx = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
        points = points[idx]
        values = values[idx]
        st.info(f"Down-sampled to {MAX_POINTS:,} points.")

    # Plotly 3D scatter
    fig = go.Figure(
        data=go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=values,
                colorscale="Hot",
                colorbar=dict(title=temp_field or "Z"),
                opacity=0.7,
            ),
        )
    )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True, key="plotly_preview")

# --------------------------------------------------------------
# 5. Convert + Split
# --------------------------------------------------------------
split_parts = st.checkbox("Split into ≤25 MiB parts (GitHub-safe)", value=True, key="split_checkbox")
max_mb = st.slider("Max part size (MiB)", 5, 25, 20, key="max_mb_slider") if split_parts else 200
MAX_BYTES = max_mb * 1024 * 1024

if st.button("Convert to .pt", type="primary", key="convert_btn"):
    import pyvista as pv
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
            all_pt_files.append(part_file)

        progress_bar.progress((idx + 1) / len(selected_names))

    status_text.success(f"Converted {len(selected_names)} simulation(s)!")
    st.balloons()

    # Download
    st.subheader("Download .pt Files")
    for pt_file in all_pt_files:
        size_mb = pt_file.stat().st_size / (1024 * 1024)
        with open(pt_file, "rb") as f:
            st.download_button(
                label=f"{pt_file.relative_to(OUTPUT_ROOT)} ({size_mb:.1f} MB)",
                data=f,
                file_name=pt_file.name,
                mime="application/octet-stream",
                key=f"dl_{pt_file.name}"
            )
    st.info(f"All files saved in: `{OUTPUT_ROOT}`")

# --------------------------------------------------------------
# 7. Reconstruction helper
# --------------------------------------------------------------
with st.expander("Re-assemble a full .pt from parts"):
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
            meta = {k: v for k, v in d.items() if not isinstance(v, torch.Tensor)}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                tensors.setdefault(k, []).append(v)
    full = {k: torch.cat(v) for k, v in tensors.items()}
    full.update(meta)
    return full
        """,
        language="python",
    )
