# --------------------------------------------------------------
# app.py – VTU → PT with Plotly 3D (No Xvfb, No OSMesa)
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
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT", layout="wide")
st.title("VTU → PyTorch (.pt) + Plotly 3D")
st.markdown("**No GPU/X11 needed** | Interactive 3D | ≤25 MiB split")

# --------------------------------------------------------------
# 2. DATA_ROOT
# --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "laser_simulations"

if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`\n\nUpload **laser_simulations/** next to `app.py`.")
    st.stop()

# --------------------------------------------------------------
# 3. Find simulations
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
# 4. Select + Plotly 3D Preview
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

# --- Read VTU with meshio (no PyVista!) ---
try:
    import meshio
    mesh = meshio.read(vtu_sample)
    points = mesh.points
    cells = mesh.cells_dict.get("tetra", None) or mesh.cells_dict.get("triangle", None)
    HAS_MESHIO = True
except ImportError:
    st.error("Install `meshio` to read .vtu files.")
    HAS_MESHIO = False

if HAS_MESHIO:
    # Downsample if too big
    if len(points) > 50_000:
        idx = np.random.choice(len(points), 50_000, replace=False)
        points = points[idx]
        st.info(f"Downsampled to 50k points for smooth rendering.")

    # Extract temperature
    temp_field = None
    for k, v in mesh.point_data.items():
        if "temp" in k.lower() or k.lower() in ("t", "temperature"):
            temp_field = k
            break
    if temp_field is None and mesh.point_data:
        temp_field = list(mesh.point_data.keys())[0]

    values = mesh.point_data.get(temp_field, points[:, 2])
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)

    # --- Plotly 3D Scatter ---
    fig = go.Figure(data=go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=values,
            colorscale='Hot',
            colorbar=dict(title=temp_field or "Z"),
            opacity=0.8
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------
# 5. Convert Button
# --------------------------------------------------------------
if st.button("Convert to .pt", type="primary"):
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
        m = meshio.read(vtu_path)
        time_val = m.field_data.get("TimeValue", [np.nan])[0] if "TimeValue" in m.field_data else np.nan
        pts = m.points

        base_data = {
            "simulation": selected,
            "P_W": sim["P"],
            "Vscan_mm_s": sim["V"],
            "time": time_val,
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
        }
        for k, arr in m.point_data.items():
            if arr.ndim == 1:
                base_data[k] = arr
            else:
                for j in range(arr.shape[1]):
                    base_data[f"{k}_{j}"] = arr[:, j]
        frames.append(pd.DataFrame(base_data))
        progress.progress((i + 1) / len(vtu_files))

    status.text("Converting to tensors...")
    full_df = pd.concat(frames, ignore_index=True)
    tensors = {c: torch.from_numpy(full_df[c].values.astype(np.float32)) 
               for c in full_df.select_dtypes(include=[np.number]).columns}
    metadata = {"simulation": selected, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}

    # Split ≤25 MiB
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
    status.success(f"Done! {len(part_files)} parts.")
    st.balloons()

    # Download
    st.subheader("Download .pt Parts")
    for f in part_files:
        with open(f, "rb") as fp:
            st.download_button(
                label=f"{f.name} ({f.stat().st_size / 1e6:.1f} MB)",
                data=fp,
                file_name=f.name,
                mime="application/octet-stream"
            )
