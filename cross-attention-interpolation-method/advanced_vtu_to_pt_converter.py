# --------------------------------------------------------------
# app.py – VTU → Full PT + Cloud SQLite + Plotly 3D
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

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → Full PT + SQLite", layout="wide")
st.title("VTU → Full PyTorch (.pt) + Cloud SQLite")
st.markdown("Convert `.vtu` → **full `.pt`** → **store in cloud SQLite** → **visualize any saved PT**")

# --------------------------------------------------------------
# RESET BUTTON
# --------------------------------------------------------------
if st.button("Reset App", help="Clear cache & GUI"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.success("Reset complete!")
    st.experimental_rerun()

# --------------------------------------------------------------
# 1. DATA_ROOT
# --------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = SCRIPT_DIR / "laser_simulations"

if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`\n\nUpload **laser_simulations/** next to `app.py`.")
    st.stop()

# --------------------------------------------------------------
# 2. Cloud SQLite (singleton – shared across users)
# --------------------------------------------------------------
@st.experimental_singleton
def get_db():
    db_path = SCRIPT_DIR / "vtu_database.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

db = get_db()

# Create table if not exists
db.execute("""
CREATE TABLE IF NOT EXISTS simulations (
    name TEXT PRIMARY KEY,
    P_W REAL,
    Vscan_mm_s REAL,
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
# 4. Select & Preview + Convert
# --------------------------------------------------------------
selected = st.selectbox(
    "Select simulation",
    options=[s["name"] for s in simulations],
    format_func=lambda x: f"{x} | P={next(s for s in simulations if s['name']==x)['P']}W, V={next(s for s in simulations if s['name']==x)['V']}mm/s",
    key="select_sim"
)

sim = next(s for s in simulations if s["name"] == selected)
vtu_sample = sorted(sim["path"].glob("*.vtu"))[0]

st.write(f"**3D Preview**: `{vtu_sample.name}`")

# Load with meshio (no VTK)
@st.cache_data
def load_preview(_path):
    import meshio
    return meshio.read(_path)

mesh = load_preview(vtu_sample)

temp_field = next((k for k in mesh.point_data if "temp" in k.lower()), None)
if temp_field is None and mesh.point_data:
    temp_field = list(mesh.point_data.keys())[0]

points = np.asarray(mesh.points)
values = np.asarray(mesh.point_data.get(temp_field, points[:, 2]))
if values.ndim > 1:
    values = np.linalg.norm(values, axis=1)

if points.shape[0] > 100_000:
    idx = np.random.choice(points.shape[0], 100_000, replace=False)
    points = points[idx]
    values = values[idx]

fig = go.Figure(data=go.Scatter3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    mode='markers',
    marker=dict(size=2, color=values, colorscale='Hot', opacity=0.7)
))
fig.update_layout(scene=dict(aspectmode='data'), height=600)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------
# 5. Convert & Save to SQLite
# --------------------------------------------------------------
if st.button("Convert & Save to Cloud SQLite", type="primary"):
    with st.spinner("Converting..."):
        import pyvista as pv
        vtu_files = sorted(sim["path"].glob("*.vtu"))
        frames = []

        for vtu_path in tqdm(vtu_files, desc="Reading"):
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

        df = pd.concat(frames, ignore_index=True)
        tensors = {c: torch.from_numpy(df[c].values.astype(np.float32)) for c in df.select_dtypes(include=[np.number]).columns}
        metadata = {"simulation": selected, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}
        full_pt = {**tensors, **metadata}

        # Save to SQLite as BLOB
        pt_bytes = io.BytesIO()
        torch.save(full_pt, pt_bytes)
        pt_bytes.seek(0)

        db.execute(
            "INSERT OR REPLACE INTO simulations (name, P_W, Vscan_mm_s, pt_data) VALUES (?, ?, ?, ?)",
            (selected, sim["P"], sim["V"], pt_bytes.read())
        )
        db.commit()

    st.success(f"Saved `{selected}.pt` to cloud SQLite!")
    st.balloons()

# --------------------------------------------------------------
# 6. Load & Visualize from SQLite
# --------------------------------------------------------------
st.subheader("Load & Visualize Saved PT from SQLite")

saved_sims = db.execute("SELECT name FROM simulations").fetchall()
saved_names = [row[0] for row in saved_sims]

if saved_names:
    load_name = st.selectbox("Select saved simulation", saved_names, key="load_select")
    if st.button("Load from SQLite"):
        row = db.execute("SELECT pt_data FROM simulations WHERE name = ?", (load_name,)).fetchone()
        pt_bytes = io.BytesIO(row[0])
        data = torch.load(pt_bytes)

        st.write(f"Loaded `{load_name}` – {data['Temperature'].shape[0]:,} points")

        # 3D Plotly
        points = np.stack([data["x"], data["y"], data["z"]], axis=1).numpy()
        values = data.get("Temperature", data["z"]).numpy()

        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color=values, colorscale='Hot', opacity=0.7)
        ))
        fig.update_layout(scene=dict(aspectmode='data'), height=600)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No simulations saved yet.")

# --------------------------------------------------------------
# 7. Download from SQLite
# --------------------------------------------------------------
if saved_names and load_name:
    row = db.execute("SELECT pt_data FROM simulations WHERE name = ?", (load_name,)).fetchone()
    st.download_button(
        label=f"Download {load_name}.pt",
        data=row[0],
        file_name=f"{load_name}.pt",
        mime="application/octet-stream"
    )
