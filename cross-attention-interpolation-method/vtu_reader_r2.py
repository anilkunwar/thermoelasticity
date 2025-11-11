# --------------------------------------------------------------
# app.py – VTU → Full PT + Cloud SQLite + stpyvista 3D
# --------------------------------------------------------------
import os
import re
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from tqdm import tqdm

st.set_page_config(page_title="VTU → Full PT + SQLite", layout="wide")
st.title("VTU → Full PyTorch (.pt) + Cloud SQLite")
st.markdown("Convert `.vtu` → **full `.pt`** → **store in cloud SQLite** → **visualize any saved PT**")

# --------------------------------------------------------------
# RESET BUTTON
# --------------------------------------------------------------
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Reset", help="Clear cache & reset GUI"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.success("Reset complete!")
        st.rerun()

# --------------------------------------------------------------
# 1. DATA_ROOT
# --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_ROOT = SCRIPT_DIR / "laser_simulations"

if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`\n\nUpload **laser_simulations/** next to `app.py`.")
    st.stop()

# --------------------------------------------------------------
# 2. Cloud SQLite
# --------------------------------------------------------------
@st.experimental_singleton
def get_db():
    db_path = SCRIPT_DIR / "laser_vtu.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

db = get_db()
db.execute("""
CREATE TABLE IF NOT EXISTS simulations (
    name TEXT PRIMARY KEY,
    P_W REAL,
    V_mm_s REAL,
    pt_data BLOB,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
db.commit()

# --------------------------------------------------------------
# 3. Detect simulations
# --------------------------------------------------------------
@st.cache_data
def find_simulations():
    pattern = re.compile(r"^P(\d+\.?\d*)_V(\d+\.?\d*)$", re.I)
    sims = []
    for p in DATA_ROOT.iterdir():
        if p.is_dir() and pattern.match(p.name):
            vtus = list(p.glob("*.vtu"))
            if vtus:
                P = float(pattern.match(p.name).group(1))
                V = float(pattern.match(p.name).group(2))
                sims.append({"name": p.name, "path": p, "P": P, "V": V, "files": len(vtus)})
    return sorted(sims, key=lambda x: x["name"])

simulations = find_simulations()
if not simulations:
    st.error("No Pxx_Vyy folders found!")
    st.stop()

st.success(f"Found {len(simulations)} simulations")

# --------------------------------------------------------------
# 4. Select simulation
# --------------------------------------------------------------
selected = st.selectbox(
    "Select simulation",
    options=[s["name"] for s in simulations],
    format_func=lambda x: f"{x} | P={next(s for s in simulations if s['name']==x)['P']}W, V={next(s for s in simulations if s['name']==x)['V']}mm/s"
)

sim = next(s for s in simulations if s["name"] == selected)
vtu_sample = sorted(sim["path"].glob("*.vtu"))[0]

st.write(f"**3D Preview**: `{vtu_sample.name}`")

# Interactive 3D with stpyvista
try:
    from stpyvista import stpyvista
    HAS_STPYVISTA = True
except ImportError:
    HAS_STPYVISTA = False
    st.warning("`stpyvista` not installed. Install with `pip install stpyvista`.")

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

if HAS_STPYVISTA:
    stpyvista(
        mesh,
        panel_kwargs=dict(orientation_widget=True, background="white", zoom=1.5),
        use_container_width=True,
        key="stpyvista_preview"
    )
else:
    # Fallback to Plotly scatter
    points = mesh.points
    values = mesh.point_data.active_scalars
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)

    fig = go.Figure(data=go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(size=2, color=values, colorscale='Hot', opacity=0.7)
    ))
    fig.update_layout(scene_aspectmode='data')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------
# 5. Convert to Attention-Ready .pt
# --------------------------------------------------------------
if st.button("Convert & Save for Attention"):
    with st.spinner("Converting..."):
        vtu_files = sorted(sim["path"].glob("*.vtu"))
        all_coords = []
        all_temp = []
        all_stress = []
        all_extra = {}  # For other fields
        times = []

        for vtu_path in tqdm(vtu_files, desc="Reading"):
            mesh = pv.read(vtu_path)
            t = mesh.field_data.get("TimeValue", [0.0])[0]
            pts = mesh.points.astype(np.float32)

            temp = mesh.point_data.get("Temperature", np.zeros(len(pts), dtype=np.float32))
            stress = mesh.point_data.get("vonmises", np.zeros(len(pts), dtype=np.float32))
            if temp.ndim > 1: temp = np.linalg.norm(temp, axis=1).astype(np.float32)
            if stress.ndim > 1: stress = np.linalg.norm(stress, axis=1).astype(np.float32)

            all_coords.append(pts)
            all_temp.append(temp)
            all_stress.append(stress)
            times.append(t)

        # Stack to tensors
        coords = torch.tensor(np.stack(all_coords), dtype=torch.float32)  # [T, N, 3]
        temperature = torch.tensor(np.stack(all_temp), dtype=torch.float32)  # [T, N]
        stress_vm = torch.tensor(np.stack(all_stress), dtype=torch.float32)  # [T, N]
        times_t = torch.tensor(times, dtype=torch.float32)

        # Compatible with your code
        solution = {
            'params': {'P_W': sim["P"], 'V_mm_s': sim["V"], 'Lx': 1.0, 't_max': times[-1]},
            'X': coords[:, :, 0],  # [T, N]
            'Y': coords[:, :, 1],  # [T, N]
            'c1_preds': temperature,  # Temperature as c1
            'c2_preds': stress_vm,  # von Mises as c2
            'times': times_t,
            'interpolated': False,
            'diffusion_type': 'laser_heating'
        }

        # Save to SQLite
        buf = io.BytesIO()
        torch.save(solution, buf)
        buf.seek(0)

        db.execute("INSERT OR REPLACE INTO simulations (name, P_W, V_mm_s, pt_data) VALUES (?, ?, ?, ?)",
                   (sim["name"], sim["P"], sim["V"], buf.read()))
        db.commit()

    st.success(f"**{sim['name']}** converted & saved to SQLite!")
    st.balloons()

# --------------------------------------------------------------
# 6. Load from SQLite & Visualize
# --------------------------------------------------------------
st.subheader("Load from SQLite")
saved = db.execute("SELECT name FROM simulations").fetchall()
if saved:
    load_name = st.selectbox("Select", [r[0] for r in saved])
    if st.button("Load & Visualize"):
        row = db.execute("SELECT pt_data FROM simulations WHERE name = ?", (load_name,)).fetchone()
        data = torch.load(io.BytesIO(row[0]))

        st.write(f"Loaded {load_name} – {data['c1_preds'].shape[0]} times")

        t_idx = st.slider("Time", 0, data['c1_preds'].shape[0]-1, 0)

        points = np.stack([data['X'][t_idx], data['Y'][t_idx], np.zeros(data['X'].shape[1])], axis=1)
        values = data['c1_preds'][t_idx].numpy()

        fig = go.Figure(data=go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],
                                          mode='markers', marker=dict(size=2, color=values, colorscale='Hot')))
        fig.update_layout(scene_aspectmode='data')
        st.plotly_chart(fig)
else:
    st.info("No simulations saved yet.")
