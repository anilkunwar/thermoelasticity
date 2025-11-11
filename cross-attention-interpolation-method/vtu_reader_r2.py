import streamlit as st
import pyvista as pv
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import os

st.title("VTU File Reader")

# Assume the directory is mounted or local in Streamlit Cloud (upload or git clone in app)
# For GitHub, you can clone the repo in the app if needed, but for simplicity, assume it's local
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

# Select folder
selected_folder = st.selectbox("Select PXX_VYY folder", folders)

if selected_folder:
    folder_path = DATA_ROOT / selected_folder
    vtu_files = sorted(folder_path.glob("*.vtu"))
    
    if not vtu_files:
        st.warning("No .vtu files in selected folder.")
        st.stop()
    
    st.success(f"Found {len(vtu_files)} .vtu files in {selected_folder}")
    
    # Select VTU file
    selected_vtu = st.selectbox("Select .vtu file", [f.name for f in vtu_files])
    
    if selected_vtu:
        vtu_path = folder_path / selected_vtu
        
        # Read VTU
        try:
            mesh = pv.read(vtu_path)
            st.header("General Information")
            st.write(f"Number of points: {mesh.n_points:,}")
            st.write(f"Number of cells: {mesh.n_cells:,}")
            st.write(f"Bounds: {mesh.bounds}")
            
            # Point data
            st.header("Point Data Fields")
            point_data = mesh.point_data
            if point_data:
                for key in point_data.keys():
                    arr = point_data[key]
                    st.subheader(key)
                    st.write(f"Shape: {arr.shape}")
                    st.write(f"Data type: {arr.dtype}")
                    st.write(f"Min value: {np.min(arr)}")
                    st.write(f"Max value: {np.max(arr)}")
                    st.write("---")
            else:
                st.write("No point data fields found.")
            
            # Cell data
            st.header("Cell Data Fields")
            cell_data = mesh.cell_data
            if cell_data:
                for key in cell_data.keys():
                    arr = cell_data[key]
                    st.subheader(key)
                    st.write(f"Shape: {arr.shape}")
                    st.write(f"Data type: {arr.dtype}")
                    st.write(f"Min value: {np.min(arr)}")
                    st.write(f"Max value: {np.max(arr)}")
                    st.write("---")
            else:
                st.write("No cell data fields found.")
            
            # Field data
            st.header("Field Data")
            field_data = mesh.field_data
            if field_data:
                for key in field_data.keys():
                    arr = field_data[key]
                    st.subheader(key)
                    st.write(f"Value: {arr}")
                    st.write("---")
            else:
                st.write("No field data found.")
            
            # 3D Preview with Plotly
            st.header("3D Preview")
            points = mesh.points
            color_field = st.selectbox("Color by field", list(mesh.point_data.keys()) or ["Z-coordinate"])
            
            if color_field == "Z-coordinate":
                colors = points[:, 2]
            else:
                colors = mesh.point_data[color_field]
                if colors.ndim > 1:
                    colors = np.linalg.norm(colors, axis=1)
            
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:,0], y=points[:,1], z=points[:,2],
                mode='markers',
                marker=dict(size=3, color=colors, colorscale='Viridis')
            )])
            fig.update_layout(scene_aspectmode='data')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading VTU: {e}")
else:
    st.info("Select a folder to see VTU files.")
