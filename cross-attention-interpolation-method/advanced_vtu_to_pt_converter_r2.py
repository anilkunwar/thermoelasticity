# --------------------------------------------------------------
# app.py – VTU → PT Converter (Streamlit Cloud + Local Compatible) - IMPROVED
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
            f"**{sim['name']}** \nP = `{sim['P']}` W  \nV = `{sim['V']}` mm/s  \nFiles: `{sim['files']}`"
        )

# --------------------------------------------------------------
# 4. Enhanced 3D Rendering Functions (Streamlit Cloud Safe)
# --------------------------------------------------------------
@st.cache_data
def load_preview_mesh(p):
    """
    Load mesh with robust error handling.
    Returns mesh object, list of point_data keys, and list of field_data keys.
    """
    try:
        # PyVista approach
        mesh = pv.read(p)
        all_point_keys = list(mesh.point_data.keys())
        all_field_keys = list(mesh.field_data.keys())
        return mesh, all_point_keys, all_field_keys
    except Exception as e:
        # st.warning(f"PyVista failed to read {p.name}: {e}")
        try:
            # Meshio fallback approach
            import meshio
            m = meshio.read(str(p))
            
            class SimpleMesh:
                pass
            sm = SimpleMesh()
            sm.points = m.points
            sm.point_data = getattr(m, "point_data", {})
            sm.field_data = getattr(m, "field_data", {})
            
            all_point_keys = list(sm.point_data.keys())
            all_field_keys = list(sm.field_data.keys())
            
            return sm, all_point_keys, all_field_keys
        except Exception as e2:
            st.error(f"Meshio also failed to read {p.name}: {e2}")
            return None, [], []

def get_plot_data(mesh, field_name):
    """Extracts point coordinates and the selected data field values."""
    pts = np.asarray(mesh.points)
    vals = None
    
    if field_name and hasattr(mesh, 'point_data') and (field_name in mesh.point_data):
        vals = np.asarray(mesh.point_data[field_name])
        if vals.ndim > 1:
            vals = np.linalg.norm(vals, axis=1) # Vector to scalar magnitude
        color_label = field_name
    else:
        # Fallback to Z-coordinate if field is not found or empty
        vals = pts[:, 2] 
        color_label = "Z-coordinate (Fallback)"
        
    return pts, vals, color_label

def calculate_clim(vals):
    """Calculates min/max for color scale using 5th and 95th percentiles."""
    if vals is None or len(vals) == 0:
        return 0, 1
        
    # Use 5th and 95th percentile to avoid color clipping due to extreme outliers
    if len(vals) > 100 and np.std(vals) > 1e-6:
        min_val = np.percentile(vals, 5)
        max_val = np.percentile(vals, 95)
    else:
        min_val = np.min(vals)
        max_val = np.max(vals)
        
    if min_val == max_val:
        min_val = min_val - 1e-6
        max_val = max_val + 1e-6
        
    return min_val, max_val

def create_plotly_3d(mesh, field_name, mesh_name):
    """Create interactive 3D plot using Plotly - IMPROVED"""
    try:
        pts, vals, color_label = get_plot_data(mesh, field_name)
        min_val, max_val = calculate_clim(vals)

        point_count = len(pts)
        # Dynamic marker size: smaller for denser point clouds
        marker_size = max(0.5, 30 / np.log10(point_count) if point_count > 10 else 2) 

        # Create Plotly 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size, 
                color=vals,
                cmin=min_val,      
                cmax=max_val,
                colorscale='Viridis', # Perceptually uniform colormap
                colorbar=dict(title=color_label, len=0.8),
                opacity=0.8
            ),
            hovertemplate='<b>X</b>: %{x:.3f}<br><b>Y</b>: %{y:.3f}<br><b>Z</b>: %{z:.3f}<br><b>' + color_label + '</b>: %{marker.color:.3f}<extra></extra>'
        )])

        fig.update_layout(
            title=dict(
                text=f"3D Preview: {mesh_name} | Field: **{color_label}**",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title='X (mm)', 
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data' # Ensures correct aspect ratio
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700 
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Plotly 3D rendering failed: {e}")
        return None

def create_2d_projection(mesh, field_name, mesh_name):
    """Create 2D projection fallback - IMPROVED Matplotlib Quality"""
    try:
        pts, vals, color_label = get_plot_data(mesh, field_name)
        min_val, max_val = calculate_clim(vals)

        # Global Matplotlib settings for high quality
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})
        
        point_count = len(pts)
        # Dynamic scatter size: smaller for denser point clouds
        scatter_s = max(0.1, 50 / np.log10(point_count) if point_count > 100 else 1.0)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY projection
        sc1 = axes[0].scatter(pts[:, 0], pts[:, 1], c=vals, s=scatter_s, cmap="viridis", vmin=min_val, vmax=max_val)
        axes[0].set_aspect("equal", adjustable='box') # Crucial for 2D geometry
        axes[0].set_title("XY Projection (Top View)")
        axes[0].set_xlabel("X (mm)")
        axes[0].set_ylabel("Y (mm)")
        cbar1 = fig.colorbar(sc1, ax=axes[0])
        cbar1.set_label(color_label)
        
        # XZ projection 
        sc2 = axes[1].scatter(pts[:, 0], pts[:, 2], c=vals, s=scatter_s, cmap="viridis", vmin=min_val, vmax=max_val)
        axes[1].set_aspect("equal", adjustable='box')
        axes[1].set_title("XZ Projection (Side View)")
        axes[1].set_xlabel("X (mm)")
        axes[1].set_ylabel("Z (mm)")
        cbar2 = fig.colorbar(sc2, ax=axes[1])
        cbar2.set_label(color_label)

        # YZ projection
        sc3 = axes[2].scatter(pts[:, 1], pts[:, 2], c=vals, s=scatter_s, cmap="viridis", vmin=min_val, vmax=max_val)
        axes[2].set_aspect("equal", adjustable='box')
        axes[2].set_title("YZ Projection (Front View)")
        axes[2].set_xlabel("Y (mm)")
        axes[2].set_ylabel("Z (mm)")
        cbar3 = fig.colorbar(sc3, ax=axes[2])
        cbar3.set_label(color_label)
        
        plt.suptitle(f"2D Projections: {mesh_name} | Field: {color_label}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight') # High resolution
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except Exception as e:
        st.error(f"2D projection failed: {e}")
        return None

def create_simple_3d_matplotlib(mesh, field_name, mesh_name):
    """Create 3D-like visualization using matplotlib - IMPROVED"""
    try:
        pts, vals, color_label = get_plot_data(mesh, field_name)
        min_val, max_val = calculate_clim(vals)

        # Global Matplotlib settings for high quality
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})
        
        point_count = len(pts)
        scatter_s = max(0.1, 50 / np.log10(point_count) if point_count > 100 else 1.0)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points with color mapping
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                        c=vals, cmap='viridis', s=scatter_s, alpha=0.7,
                        vmin=min_val, vmax=max_val)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)') 
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Point Cloud: {mesh_name} | Field: {color_label}')
        
        # Set equal-like aspect ratio (manual scaling)
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), 
                              pts[:,1].max()-pts[:,1].min(), 
                              pts[:,2].max()-pts[:,2].min()]).max() / 2.0

        mid_x = (pts[:,0].max()+pts[:,0].min()) / 2.0
        mid_y = (pts[:,1].max()+pts[:,1].min()) / 2.0
        mid_z = (pts[:,2].max()+pts[:,2].min()) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label(color_label)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except Exception as e:
        st.error(f"Matplotlib 3D also failed: {e}")
        return None

# --------------------------------------------------------------
# 5. Select & Preview with Streamlit-Safe 3D Rendering - ENHANCED
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
        # Time step selector for multi-file simulations
        vtu_sample_idx = 0
        if len(vtu_files) > 1:
            vtu_sample_idx = st.slider("Select time step for preview", 0, len(vtu_files)-1, 0)
        
        vtu_sample = vtu_files[vtu_sample_idx]
        st.subheader(f"3-D Preview: `{vtu_sample.name}`")
        st.caption(f"(Step {vtu_sample_idx+1}/{len(vtu_files)})")

        # Load mesh and retrieve all field keys
        mesh, point_keys, field_keys = load_preview_mesh(vtu_sample)
        
        if mesh is None:
            st.warning("Preview unavailable (could not load mesh).")
        else:
            # 1. Field Data Display (Global Metadata)
            if field_keys:
                field_info = "\n".join([f"**{k}**: `{mesh.field_data.get(k, ['N/A'])[0]}`" for k in field_keys])
                st.info(f"**Global Field Data (Metadata):**\n{field_info}")

            # 2. Point Data Selection (For Coloring)
            available_fields = point_keys
            
            # Smart default field selection (Temperature > T > first key)
            default_field = None
            for k in available_fields:
                if "temp" in k.lower() or k.lower() in ("t", "temperature"):
                    default_field = k
                    break
            if default_field is None and available_fields:
                default_field = available_fields[0]
                
            selected_field = st.selectbox(
                "Select Data Field to Visualize (Point Data)",
                options=available_fields,
                index=available_fields.index(default_field) if default_field in available_fields else 0,
                help="Choose the point-associated data (e.g., Temperature, Velocity) to color the mesh points."
            )

            # 3. Rendering Method Selector 
            options = ["2D Projections", "Matplotlib 3D"]
            if PLOTLY_AVAILABLE:
                options.insert(0, "Plotly 3D (Interactive)")
                index = 0
            else:
                index = 0
                
            render_method = st.radio(
                "Rendering Method",
                options,
                index=index,
                help="Plotly 3D works best in Streamlit Cloud. 2D Projections offer static high-res images."
            )

            # 4. Rendering Execution (Using selected_field)
            if selected_field:
                if render_method == "Plotly 3D (Interactive)" and PLOTLY_AVAILABLE:
                    fig = create_plotly_3d(mesh, selected_field, vtu_sample.name)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Plotly 3D rendering failed, falling back to 2D projections")
                        # Fallback rendering...
                
                elif render_method == "2D Projections":
                    buf = create_2d_projection(mesh, selected_field, vtu_sample.name)
                    if buf:
                        st.image(buf, use_column_width=True, caption=f"2D Projections (Field: {selected_field})")
                        
                elif render_method == "Matplotlib 3D":
                    buf = create_simple_3d_matplotlib(mesh, selected_field, vtu_sample.name)
                    if buf:
                        st.image(buf, use_column_width=True, caption=f"3D Matplotlib Point Cloud (Field: {selected_field})")
            
            # Show mesh statistics
            if hasattr(mesh, 'points'):
                st.write("**Mesh Statistics:**")
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Points", f"{len(mesh.points):,}")
                with cols[1]:
                    # PyVista mesh has n_cells, SimpleMesh does not
                    if hasattr(mesh, 'n_cells'):
                        st.metric("Cells", f"{mesh.n_cells:,}")
                    else:
                        st.metric("Cells", "N/A (Simple Mesh)")
                with cols[2]:
                    bounds = np.ptp(mesh.points, axis=0)
                    st.metric("Size X", f"{bounds[0]:.3f} mm")
                with cols[3]:
                    st.metric("Size Y", f"{bounds[1]:.3f} mm")

# --------------------------------------------------------------
# 6. Convert (optional split)
# --------------------------------------------------------------
st.markdown("---")
st.header("Conversion to PyTorch (.pt)")
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
            # Using PyVista for conversion as it's more robust than SimpleMesh
            try:
                mesh = pv.read(vtu_path)
            except Exception as e:
                st.warning(f"Skipping {vtu_path.name} due to read error: {e}")
                continue
                
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

        if not frames:
             st.warning(f"No data successfully extracted for simulation {name}.")
             continue
             
        full_df = pd.concat(frames, ignore_index=True)
        numeric_cols = full_df.select_dtypes(include=[np.number]).columns

        tensors = {
            col: torch.from_numpy(full_df[col].values.astype(np.float32))
            for col in numeric_cols
        }
        metadata = {"simulation": name, "P_W": float(sim["P"]), "Vscan_mm_s": float(sim["V"])}

        # ---------- Split ----------
        N = next(iter(tensors.values())).shape[0]
        # Estimate row bytes based on float32 (4 bytes per element)
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

    status_text.success(f"Converted {len(selected_names)} simulation(s)! Total files: {len(all_pt_files)}")
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
    if not parts:
        raise FileNotFoundError(f"No part files found in {folder}")

    tensors = {}
    meta = None
    
    for p in parts:
        d = torch.load(p)
        # Store metadata from the first part
        if meta is None:
            # Filter out tensors and internal fields like part_index, total_parts
            meta_keys = [k for k, v in d.items() if not isinstance(v, torch.Tensor) and k not in ('part_index', 'total_parts')]
            meta = {k: d[k] for k in meta_keys}
            
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and k not in ('part_index', 'total_parts'):
                tensors.setdefault(k, []).append(v)
                
    # Concatenate all tensors
    full = {k: torch.cat(v) for k, v in tensors.items()}
    full.update(meta)
    return full
        """,
        language="python",
    )
