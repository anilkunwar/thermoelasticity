# --------------------------------------------------------------
# app.py – GUI .vtu → .pt Converter (Streamlit + 3D Preview)
# --------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import glob
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# 1. HEADLESS RENDERING – MUST BE BEFORE pyvista import!
# ==============================================================
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_AUTO_CLOSE"] = "False"
os.environ["PYVISTA_USE_PANEL"] = "True"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import pyvista as pv
try:
    pv.start_xvfb()
except:
    pass

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="VTU → PT Converter", layout="wide")
st.title("VTU to PyTorch (.pt) Converter")
st.markdown("Convert laser heating `.vtu` files → ML-ready `.pt` tensors with **3D preview**.")

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
        "Upload a folder named **laser_simulations** (with sub-folders P10_V65, …) "
        "next to `app.py`."
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
                sims.append(
                    {
                        "name": p.name,
                        "path": p,
                        "P": P,
                        "V": V,
                        "files": len(vtu_files),
                    }
                )
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
# 4. Select & 3-D preview (robust)
# --------------------------------------------------------------
selected_names = st.multiselect(
    "Select simulations to convert",
    options=[s["name"] for s in simulations],
    default=[s["name"] for s in simulations[:1]],
)

if selected_names:
    first_sim = next(s for s in simulations if s["name"] == selected_names[0])
    vtu_sample = sorted(first_sim["path"].glob("*.vtu"))[0]
    st.write(f"**3-D Preview**: `{vtu_sample.name}`")

    # ---------- Load mesh (pyvista → fallback meshio) ----------
    @st.cache_data
    def load_mesh(path):
        try:
            return pv.read(path)
        except Exception:
            try:
                import meshio
                m = meshio.read(str(path))
                class SimpleMesh:
                    pass
                sm = SimpleMesh()
                sm.points = m.points
                sm.point_data = getattr(m, "point_data", {})
                sm.field_data = getattr(m, "field_data", {})
                return sm
            except Exception as e:
                st.error(f"Cannot read mesh for preview: {e}")
                return None

    mesh = load_mesh(vtu_sample)
    if mesh is None:
        st.warning("Preview unavailable – mesh could not be loaded.")
    else:
        # ---------- Choose scalar field ----------
        try:
            keys = list(mesh.point_data.keys())
        except Exception:
            keys = []
        temp_field = None
        for k in keys:
            if "temp" in k.lower() or k.lower() in ("t", "temperature"):
                temp_field = k
                break
        if temp_field is None and keys:
            temp_field = keys[0]

        # ---------- 3-D rendering (pyvista) ----------
        preview_3d = None
        try:
            plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
            plotter.add_mesh(
                mesh,
                scalars=temp_field,
                cmap="hot",
                show_scalar_bar=True,
                lighting=True,
            )
            plotter.set_background("white")
            plotter.camera_position = "xy"
            preview_3d = plotter.screenshot(transparent_background=True, return_img=True)
            plotter.close()
        except Exception as e:
            st.warning(f"3-D rendering failed: {e}")

        # ---------- Show 3-D if we have it ----------
        if preview_3d is not None:
            st.image(preview_3d, use_column_width=True)
        else:
            # ---------- 2-D fallback (matplotlib scatter) ----------
            try:
                pts = np.asarray(mesh.points)
                if temp_field and temp_field in getattr(mesh, "point_data", {}):
                    vals = np.asarray(mesh.point_data[temp_field])
                    if vals.ndim > 1:
                        vals = np.linalg.norm(vals, axis=1)
                else:
                    vals = pts[:, 2]  # fallback to Z

                fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
                sc = ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=vals,
                    s=4,
                    cmap="hot",
                    marker=".",
                    rasterized=True,
                )
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{vtu_sample.name} – XY projection")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                plt.colorbar(sc, ax=ax, label=temp_field or "Z")
                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close(fig)
            except Exception as e2:
                st.error(f"Even 2-D fallback failed: {e2}")

# --------------------------------------------------------------
# 5. Convert (optional split)
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
        metadata = {
            "simulation": name,
            "P_W": float(sim["P"]),
            "Vscan_mm_s": float(sim["V"]),
        }

        # ---------- FIXED LINE ----------
        N = next(iter(tensors.values())).shape[0]   # <-- correct syntax

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
    # 6. Download results
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
# 7. Reconstruction helper
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

# Example
data = load_simulation("processed_pt/P10_V65")
print(data["Temperature"].shape)
        """,
        language="python",
    )
