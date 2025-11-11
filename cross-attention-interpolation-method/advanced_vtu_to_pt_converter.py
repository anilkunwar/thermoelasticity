# --------------------------------------------------------------
# app.py – VTU → PT Converter (Streamlit Cloud + Local Compatible)
# --------------------------------------------------------------
import os
import re
import glob
import io
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# 1. STREAMLIT CLOUD SAFE CONFIGURATION (No OpenGL/X11 dependencies)
# ==============================================================
# Minimal PyVista configuration for reading only
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_USE_PANEL"] = "False"  # No Panel/OpenGL
os.environ["PYVISTA_AUTO_CLOSE"] = "True"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "True"
os.environ["VTK_USE_OFFSCREEN"] = "True"

import pyvista as pv
# DO NOT call pv.start_xvfb() - it requires X11 libraries
pv.OFF_SCREEN = True

# Import Plotly for 3D visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available for 3D visualization")

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT Converter", layout="wide")
st.title("VTU to PyTorch (.pt) Converter")
st.markdown("Convert laser-heating `.vtu` files to ML-ready `.pt` tensors **with 3-D preview**.")

# --------------------------------------------------------------
# 2. DATA_ROOT – relative to this script
# --------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = "laser_simulations"

data_folder = st.text_input(
    "Folder with Pxx_Vyy sub-folders",
    value=DEFAULT_FOLDER,
    help="Leave default if the folder is next to `app.py`."
)

DATA_ROOT = SCRIPT_DIR / data_folder
if not DATA_ROOT.exists():
    st.error(
        f"Folder not found: `{DATA_ROOT}`\n\n"
        "Make sure the folder **laser_simulations** (containing P10_V65, …) "
        "is in the **same directory** as `app.py`."
    )
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
# 4. Enhanced 3D Rendering Functions (Streamlit Cloud Safe)
# --------------------------------------------------------------
@st.cache_data
def load_preview_mesh(p):
    """Load mesh with robust error handling - PyVista for reading only"""
    try:
        return pv.read(p)
    except Exception as e:
        st.warning(f"PyVista failed to read {p.name}: {e}")
        try:
            import meshio
            m = meshio.read(str(p))
            class SimpleMesh:
                pass
            sm = SimpleMesh()
            sm.points = m.points
            sm.point_data = getattr(m, "point_data", {})
            sm.field_data = getattr(m, "field_data", {})
            return sm
        except Exception as e2:
            st.error(f"Meshio also failed to read {p.name}: {e2}")
            return None

def create_plotly_3d(mesh, temp_field, mesh_name):
    """Create interactive 3D plot using Plotly (no OpenGL dependencies)"""
    try:
        pts = np.asarray(mesh.points)
        
        # Get values for coloring
        if temp_field and hasattr(mesh, 'point_data') and (temp_field in mesh.point_data):
            vals = np.asarray(mesh.point_data[temp_field])
            if vals.ndim > 1:
                vals = np.linalg.norm(vals, axis=1)
            color_label = temp_field
        else:
            vals = pts[:, 2]  # Use Z-coordinate as fallback
            color_label = "Z-coordinate"

        # Create Plotly 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1], 
            z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=vals,
                colorscale='Hot',
                colorbar=dict(title=color_label),
                opacity=0.7
            ),
            hovertemplate='<b>X</b>: %{x:.3f}<br><b>Y</b>: %{y:.3f}<br><b>Z</b>: %{z:.3f}<br><b>Value</b>: %{marker.color:.3f}<extra></extra>'
        )])

        fig.update_layout(
            title=dict(
                text=f"3D Preview: {mesh_name}",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Plotly 3D rendering failed: {e}")
        return None

def create_2d_projection(mesh, temp_field, mesh_name):
    """Create 2D projection fallback"""
    try:
        pts = np.asarray(mesh.points)
        
        # Get values for coloring
        if temp_field and hasattr(mesh, 'point_data') and (temp_field in mesh.point_data):
            vals = np.asarray(mesh.point_data[temp_field])
            if vals.ndim > 1:
                vals = np.linalg.norm(vals, axis=1)
        else:
            vals = pts[:, 2]  # Use Z-coordinate as fallback

        # Create 2D projection
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY projection
        sc1 = axes[0].scatter(pts[:, 0], pts[:, 1], c=vals, s=1, cmap="hot")
        axes[0].set_aspect("equal")
        axes[0].set_title("XY Projection")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        plt.colorbar(sc1, ax=axes[0])
        
        # XZ projection  
        sc2 = axes[1].scatter(pts[:, 0], pts[:, 2], c=vals, s=1, cmap="hot")
        axes[1].set_aspect("equal")
        axes[1].set_title("XZ Projection")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        plt.colorbar(sc2, ax=axes[1])
        
        # YZ projection
        sc3 = axes[2].scatter(pts[:, 1], pts[:, 2], c=vals, s=1, cmap="hot")
        axes[2].set_aspect("equal")
        axes[2].set_title("YZ Projection")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")
        plt.colorbar(sc3, ax=axes[2])
        
        plt.suptitle(f"2D Projections: {mesh_name}")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except Exception as e:
        st.error(f"2D projection failed: {e}")
        return None

def create_simple_3d_matplotlib(mesh, temp_field, mesh_name):
    """Create 3D-like visualization using matplotlib"""
    try:
        pts = np.asarray(mesh.points)
        
        # Get values for coloring
        if temp_field and hasattr(mesh, 'point_data') and (temp_field in mesh.point_data):
            vals = np.asarray(mesh.point_data[temp_field])
            if vals.ndim > 1:
                vals = np.linalg.norm(vals, axis=1)
        else:
            vals = pts[:, 2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points with color mapping
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                       c=vals, cmap='hot', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'3D Point Cloud: {mesh_name}')
        
        plt.colorbar(sc, ax=ax, label=(temp_field or 'Value'))
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except Exception as e:
        st.error(f"Matplotlib 3D also failed: {e}")
        return None

# --------------------------------------------------------------
# 5. Select & Preview with Streamlit-Safe 3D Rendering
# --------------------------------------------------------------
selected_names = st.multiselect(
    "Select simulations to convert",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]],
)

if selected_names:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_files = sorted(first_sim["path"].glob("*.vtu"))
    
    if vtu_files:
        vtu_sample = vtu_files[0]
        st.write(f"**3-D Preview**: `{vtu_sample.name}`")

        # Time step selector for multi-file simulations
        if len(vtu_files) > 1:
            time_step = st.slider("Select time step", 0, len(vtu_files)-1, 0)
            vtu_sample = vtu_files[time_step]
            st.write(f"Showing: `{vtu_sample.name}` (Step {time_step+1}/{len(vtu_files)})")

        mesh = load_preview_mesh(vtu_sample)
        
        if mesh is None:
            st.warning("Preview unavailable (could not load mesh).")
        else:
            # Pick a temperature-like field
            temp_field = None
            try:
                keys = list(mesh.point_data.keys())
                st.info(f"Available fields: {keys}")
            except Exception:
                keys = []
                
            for k in keys:
                if "temp" in k.lower() or k.lower() in ("t", "temperature"):
                    temp_field = k
                    break
            if temp_field is None and keys:
                temp_field = keys[0]

            # Rendering method selector
            if PLOTLY_AVAILABLE:
                render_method = st.radio(
                    "Rendering Method",
                    ["Plotly 3D (Interactive)", "2D Projections", "Matplotlib 3D"],
                    index=0,
                    help="Plotly 3D works best in Streamlit Cloud"
                )
            else:
                render_method = st.radio(
                    "Rendering Method",
                    ["2D Projections", "Matplotlib 3D"],
                    index=0,
                    help="Plotly not available - using fallback methods"
                )

            if render_method == "Plotly 3D (Interactive)" and PLOTLY_AVAILABLE:
                fig = create_plotly_3d(mesh, temp_field, vtu_sample.name)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly 3D rendering failed, falling back to 2D projections")
                    buf = create_2d_projection(mesh, temp_field, vtu_sample.name)
                    if buf:
                        st.image(buf, use_column_width=True, caption="2D Projections (Fallback)")
                    
            elif render_method == "2D Projections":
                buf = create_2d_projection(mesh, temp_field, vtu_sample.name)
                if buf:
                    st.image(buf, use_column_width=True, caption="2D Projections")
                    
            elif render_method == "Matplotlib 3D":
                buf = create_simple_3d_matplotlib(mesh, temp_field, vtu_sample.name)
                if buf:
                    st.image(buf, use_column_width=True, caption="3D Matplotlib Point Cloud")

            # Show mesh statistics
            if hasattr(mesh, 'points'):
                st.write("**Mesh Statistics:**")
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Points", f"{len(mesh.points):,}")
                with cols[1]:
                    if hasattr(mesh, 'n_cells'):
                        st.metric("Cells", f"{mesh.n_cells:,}")
                with cols[2]:
                    bounds = np.ptp(mesh.points, axis=0)
                    st.metric("Size X", f"{bounds[0]:.1f}")
                with cols[3]:
                    st.metric("Size Y", f"{bounds[1]:.1f}")

# --------------------------------------------------------------
# 6. Convert (optional split)
# --------------------------------------------------------------
split_parts = st.checkbox("Split into ≤25 MiB parts (GitHub-safe)", value=True)
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

        # ---------- Split ----------
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

    # --------------------------------------------------------------
    # 7. Download results
    # --------------------------------------------------------------
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
    st.info(f"All files saved in: `{OUTPUT_ROOT}`")

# --------------------------------------------------------------
# 8. Reconstruction helper
# --------------------------------------------------------------
with st.expander("Re-assemble a full .pt from parts"):
    st.code(
        """
import torch, glob, os
from pathlib import Path

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
