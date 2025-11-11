#!/usr/bin/env python
# --------------------------------------------------------------
# vtu_to_pt.py
#   Convert folders of .vtu files → one .pt file per simulation
#   Folder name must be P<power>_V<speed>  (e.g. P10_V65)
# --------------------------------------------------------------

import os
import re
import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pyvista as pv
import torch
import numpy as np

# --------------------------------------------------------------
# CONFIG – change only these two lines if needed
# --------------------------------------------------------------
DATA_ROOT   = Path("laser_simulations")   # folder that contains Pxx_Vyy sub-folders
OUTPUT_ROOT = Path("processed_pt")        # where .pt files will be written
OUTPUT_ROOT.mkdir(exist_ok=True)

# --------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------
def list_simulation_folders():
    """Return sorted list of folders that match Pxx_Vyy and contain .vtu files."""
    pattern = re.compile(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", re.IGNORECASE)
    folders = []
    for p in DATA_ROOT.iterdir():
        if p.is_dir() and pattern.match(p.name) and any(p.glob("*.vtu")):
            folders.append(p)
    return sorted(folders, key=lambda x: x.name)

def parse_P_V(folder_path: Path):
    m = re.match(r"^P(\d+(?:\.\d+)?)_V(\d+(?:\.\d+)?)$", folder_path.name, re.IGNORECASE)
    return float(m.group(1)), float(m.group(2))

def load_vtu(vtu_path):
    """Read a .vtu file → dict of numpy arrays (float32)."""
    mesh = pv.read(vtu_path)
    data = {
        "points": mesh.points.astype(np.float32),
        "time": float(mesh.field_data.get("TimeValue", [np.nan])[0]),
    }
    # point data (scalar or vector → split into components)
    for name in mesh.point_data:
        arr = mesh.point_data[name]
        if arr.ndim == 1:
            data[name] = arr.astype(np.float32)
        else:                     # vector field
            for i in range(arr.shape[1]):
                data[f"{name}_{i}"] = arr[:, i].astype(np.float32)
    # cell data (prefixed with "cell_")
    for name in mesh.cell_data:
        arr = mesh.cell_data[name]
        if arr.ndim == 1:
            data[f"cell_{name}"] = arr.astype(np.float32)
    return data

def vtu_to_dataframe(vtu_dict, sim_name, P, V):
    npts = vtu_dict["points"].shape[0]
    base = {
        "simulation": sim_name,
        "P_W": P,
        "Vscan_mm_s": V,
        "time": vtu_dict["time"],
        "x": vtu_dict["points"][:, 0],
        "y": vtu_dict["points"][:, 1],
        "z": vtu_dict["points"][:, 2],
    }
    rows = []
    for k, v in vtu_dict.items():
        if k in {"points", "time"}:
            continue
        if v.shape[0] == npts:               # point data only
            row = base.copy()
            row[k] = v
            rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# --------------------------------------------------------------
# MAIN CONVERSION LOOP
# --------------------------------------------------------------
def main():
    folders = list_simulation_folders()
    if not folders:
        print(f"No Pxx_Vyy folders with .vtu files found in {DATA_ROOT}")
        return

    print(f"Found {len(folders)} simulation folders:")
    for f in folders:
        print(f"  • {f.name}")

    for folder_path in folders:
        sim_name = folder_path.name
        P, V = parse_P_V(folder_path)
        print(f"\n--- Processing {sim_name} (P={P} W, V={V} mm/s) ---")

        vtu_files = sorted(glob.glob(str(folder_path / "*.vtu")))
        if not vtu_files:
            print("  [Warning] No .vtu files – skipping")
            continue

        frames = []
        for fp in tqdm(vtu_files, desc="Reading .vtu", leave=False):
            vtu = load_vtu(fp)
            df = vtu_to_dataframe(vtu, sim_name, P, V)
            if not df.empty:
                frames.append(df)

        if not frames:
            print("  [Warning] No data extracted – skipping")
            continue

        full_df = pd.concat(frames, ignore_index=True)
        print(f"  → {len(full_df):,} rows (nodes × time steps)")

        # ---- Save as .pt -------------------------------------------------
        pt_path = OUTPUT_ROOT / f"{sim_name}.pt"
        tensor_dict = {
            col: torch.from_numpy(full_df[col].values.astype(np.float32))
            for col in full_df.columns
        }
        torch.save(tensor_dict, pt_path)
        print(f"  → Saved {pt_path} ({pt_path.stat().st_size / 1e6:.1f} MB)")

    print("\nAll done! .pt files are in:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
