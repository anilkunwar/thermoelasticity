# Streamlit App to Read VTU Files
import streamlit as st
import pyvista as pv
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

st.title("VTU File Reader")

# Upload VTU file
uploaded_file = st.file_uploader("Upload .vtu file", type="vtu")

if uploaded_file is not None:
    # Save temporarily
    temp_path = Path("temp.vtu")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Read VTU
    mesh = pv.read(temp_path)

    # Display info
    st.header("General Info")
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

    # Simple 3D preview
    st.header("3D Preview")
    points = mesh.points
    values = mesh.point_data.get('Temperature', np.zeros(len(points)))
    fig = go.Figure(data=go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(size=3, color=values, colorscale='Hot')
    ))
    st.plotly_chart(fig)

    # Cleanup
    os.remove(temp_path)
else:
    st.info("Upload a .vtu file to extract information.")
