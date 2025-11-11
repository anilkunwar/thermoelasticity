import streamlit as st
import pyvista as pv
import numpy as np
from pathlib import Path

st.title("VTU File Information Extractor")

# Upload VTU file
uploaded_file = st.file_uploader("Upload .vtu file", type="vtu")

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Path("temp.vtu")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Read VTU file
    mesh = pv.read(temp_path)

    # Extract information
    st.header("General Information")
    st.write(f"Number of points: {mesh.n_points:,}")
    st.write(f"Number of cells: {mesh.n_cells:,}")
    st.write(f"Bounds: {mesh.bounds}")

    # Point data fields
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

    # Cell data fields
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

    # Cleanup temp file
    os.remove(temp_path)
else:
    st.info("Please upload a .vtu file to extract information.")
