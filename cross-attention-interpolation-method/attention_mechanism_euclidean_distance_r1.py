# --------------------------------------------------------------
# Enhanced VTU Reader with Temporal Sequencing
# --------------------------------------------------------------
import streamlit as st
import pyvista as pv
import numpy as np
import torch
import sqlite3
import io
from pathlib import Path
from tqdm import tqdm
import os
import re
import pandas as pd
import torch.nn as nn
from scipy.interpolate import interp1d
st.set_page_config(page_title="Laser VTU Temporal Reader", layout="wide")
st.title("🧭 Laser VTU Reader with Temporal Sequencing")
st.markdown("**Pxx_Vyy → Time-aware .pt sequences for interpolation**")
# --------------------------------------------------------------
# TIMESTEP PATTERNS (from your data)
# --------------------------------------------------------------
TIMESTEP_PATTERNS = {
    "P10_V65": [1.0e-6, 1.0e-4] * 18 + [1.0e-6], # 37 steps
    "P35_V65": [1.0e-6, 1.0e-4] * 19 + [1.0e-6], # 39 steps
    "P50_V65": [1.0e-6, 1.0e-4] * 19, # 38 steps
    "P50_V80": [1.0e-6, 1.0e-4] * 23, # 46 steps
}
# --------------------------------------------------------------
# RESET
# --------------------------------------------------------------
if st.button("Reset App"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()
# --------------------------------------------------------------
# DATA_ROOT
# --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_FOLDER = "laser_simulations"
data_folder = st.text_input("Folder with Pxx_Vyy", value=DEFAULT_FOLDER)
DATA_ROOT = SCRIPT_DIR / data_folder
if not DATA_ROOT.exists():
    st.error(f"Folder not found: `{DATA_ROOT}`")
    st.stop()
# --------------------------------------------------------------
# Find simulations with automatic pattern detection
# --------------------------------------------------------------
@st.cache_data
def find_simulations():
    pattern = re.compile(r"^P(\d+)_V(\d+)$", re.I)
    sims = []
   
    for p in DATA_ROOT.iterdir():
        if p.is_dir() and pattern.match(p.name):
            vtus = sorted(p.glob("*.vtu"))
            if vtus:
                P = float(pattern.match(p.name).group(1))
                V = float(pattern.match(p.name).group(2))
               
                # Auto-detect or use predefined pattern
                if p.name in TIMESTEP_PATTERNS:
                    time_pattern = TIMESTEP_PATTERNS[p.name]
                    pattern_source = "predefined"
                else:
                    # Default pattern for unknown simulations
                    n_files = len(vtus)
                    time_pattern = [1.0e-6, 1.0e-4] * (n_files // 2)
                    if n_files % 2 == 1:
                        time_pattern.append(1.0e-6)
                    pattern_source = "auto-generated"
               
                sims.append({
                    "name": p.name,
                    "path": p,
                    "P": P,
                    "V": V,
                    "files": len(vtus),
                    "time_pattern": time_pattern,
                    "pattern_source": pattern_source,
                    "total_time": sum(time_pattern)
                })
   
    return sorted(sims, key=lambda x: (x["P"], x["V"]))
simulations = find_simulations()
if not simulations:
    st.error("No Pxx_Vyy folders found!")
    st.stop()
# Display simulation info
st.success(f"Found {len(simulations)} simulations")
# --------------------------------------------------------------
# Simulation selection with temporal info
# --------------------------------------------------------------
selected = st.selectbox(
    "Select simulation",
    options=[s["name"] for s in simulations],
    format_func=lambda x: (
        f"{x} | P={next(s for s in simulations if s['name']==x)['P']}W | "
        f"V={next(s for s in simulations if s['name']==x)['V']}mm/s | "
        f"{next(s for s in simulations if s['name']==x)['files']} steps | "
        f"{next(s for s in simulations if s['name']==x)['pattern_source']} pattern"
    )
)
sim = next(s for s in simulations if s["name"] == selected)
# --------------------------------------------------------------
# Display temporal information
# --------------------------------------------------------------
st.subheader("📊 Temporal Sequence")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total timesteps", sim["files"])
with col2:
    st.metric("Total simulation time", f"{sim['total_time']:.4f} s")
with col3:
    st.metric("Pattern source", sim["pattern_source"])
# Show timestep pattern
st.write("**Timestep sequence:**")
pattern_df = pd.DataFrame({
    'Step': range(1, len(sim["time_pattern"]) + 1),
    'Δt (s)': sim["time_pattern"],
    'Cumulative Time (s)': np.cumsum(sim["time_pattern"])
})
st.dataframe(pattern_df, use_container_width=True, height=200)
# --------------------------------------------------------------
# Enhanced VTU Reader with Temporal Data
# --------------------------------------------------------------
def read_vtu_with_temporal_data(vtu_path, time_value):
    """Read VTU file and return data with temporal context"""
    mesh = pv.read(vtu_path)
   
    # Extract all relevant fields
    points = mesh.points.astype(np.float32)
   
    # Temperature and thermal data
    temperature = mesh.point_data["temperature"].astype(np.float32)
   
    # Stress data
    stress_components = {}
    for key in ['stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'stress_yz', 'stress_xz']:
        if key in mesh.point_data:
            stress_components[key] = mesh.point_data[key].astype(np.float32)
   
    vonmises = mesh.point_data["vonmises"].astype(np.float32)
    principal_stress = mesh.point_data["principal stress"].astype(np.float32)
   
    # Strain data
    strain_components = {}
    for key in ['strain_xx', 'strain_yy', 'strain_zz', 'strain_xy', 'strain_yz', 'strain_xz']:
        if key in mesh.point_data:
            strain_components[key] = mesh.point_data[key].astype(np.float32)
   
    principal_strain = mesh.point_data["principal strain"].astype(np.float32)
   
    # Displacement and velocity
    displacement = mesh.point_data["displacement"].astype(np.float32)
    velocity = mesh.point_data["velocity"].astype(np.float32)
   
    # Pressure and loads
    pressure = mesh.point_data["pressure"].astype(np.float32)
   
    return {
        'points': points,
        'temperature': temperature,
        'stress_components': stress_components,
        'vonmises': vonmises,
        'principal_stress': principal_stress,
        'strain_components': strain_components,
        'principal_strain': principal_strain,
        'displacement': displacement,
        'velocity': velocity,
        'pressure': pressure,
        'time': time_value,
        'n_points': mesh.n_points,
        'n_cells': mesh.n_cells
    }
# --------------------------------------------------------------
# Convert to Temporal .pt Sequence
# --------------------------------------------------------------
if st.button("🔄 Convert to Temporal .pt Sequence", type="primary"):
    with st.spinner(f"Reading {sim['files']} timesteps with temporal sequencing..."):
        vtu_files = sorted(sim["path"].glob("*.vtu"))

        # ---- NEW: drop a_t0001.vtu (first file) -----------------
        if vtu_files and vtu_files[0].name == "a_t0001.vtu":
            vtu_files = vtu_files[1:]                     # <-- skip it
            # also drop the first Δt (which would belong to the empty step)
            time_pattern = sim["time_pattern"][1:]
        else:
            time_pattern = sim["time_pattern"]
        # ---------------------------------------------------------

        # Recalculate cumulative times (start at t = 0)
        cumulative_times = np.cumsum([0.0] + time_pattern)[:-1]

        all_data = []
        progress_bar = st.progress(0)

        for i, (vtu_path, time_val) in enumerate(
            tqdm(zip(vtu_files, cumulative_times),
                 total=len(vtu_files), desc="Reading VTU")
        ):
            try:
                frame_data = read_vtu_with_temporal_data(vtu_path, time_val)
                all_data.append(frame_data)
                progress_bar.progress((i + 1) / len(vtu_files))
            except Exception as e:
                st.error(f"Error reading {vtu_path.name}: {e}")
                continue
       
        if not all_data:
            st.error("No VTU files could be read!")
            st.stop()
       
        # Convert to torch tensors with temporal structure
        n_timesteps = len(all_data)
        n_points = all_data[0]['n_points']
       
        # Stack temporal sequences
        times_t = torch.tensor([d['time'] for d in all_data], dtype=torch.float32)
       
        # Coordinates [T, N, 3]
        coords = torch.stack([torch.tensor(d['points']) for d in all_data])
       
        # Scalar fields [T, N]
        temperature = torch.stack([torch.tensor(d['temperature']) for d in all_data])
        vonmises = torch.stack([torch.tensor(d['vonmises']) for d in all_data])
        pressure = torch.stack([torch.tensor(d['pressure']) for d in all_data])
       
        # Vector fields [T, N, 3]
        displacement = torch.stack([torch.tensor(d['displacement']) for d in all_data])
        velocity = torch.stack([torch.tensor(d['velocity']) for d in all_data])
        principal_stress = torch.stack([torch.tensor(d['principal_stress']) for d in all_data])
        principal_strain = torch.stack([torch.tensor(d['principal_strain']) for d in all_data])
       
        # Stress components [T, N, 6]
        stress_components = torch.stack([
            torch.stack([
                torch.tensor(d['stress_components']['stress_xx']),
                torch.tensor(d['stress_components']['stress_yy']),
                torch.tensor(d['stress_components']['stress_zz']),
                torch.tensor(d['stress_components']['stress_xy']),
                torch.tensor(d['stress_components']['stress_yz']),
                torch.tensor(d['stress_components']['stress_xz'])
            ], dim=1) for d in all_data
        ])
       
        # Strain components [T, N, 6]
        strain_components = torch.stack([
            torch.stack([
                torch.tensor(d['strain_components']['strain_xx']),
                torch.tensor(d['strain_components']['strain_yy']),
                torch.tensor(d['strain_components']['strain_zz']),
                torch.tensor(d['strain_components']['strain_xy']),
                torch.tensor(d['strain_components']['strain_yz']),
                torch.tensor(d['strain_components']['strain_xz'])
            ], dim=1) for d in all_data
        ])
       
        # Create comprehensive output structure
        temporal_solution = {
            'metadata': {
                'sim_name': sim["name"],
                'P_W': sim["P"],
                'V_mm_s': sim["V"],
                'n_timesteps': n_timesteps,
                'n_points': n_points,
                'total_time': sim["total_time"],
                'time_pattern': sim["time_pattern"],
                'interpolated': False,
                'physics_type': 'laser_thermoelasticity_temporal'
            },
            'temporal_data': {
                'times': times_t, # [T]
                'coordinates': coords, # [T, N, 3]
                'temperature': temperature, # [T, N]
                'vonmises_stress': vonmises, # [T, N]
                'pressure': pressure, # [T, N]
                'displacement': displacement, # [T, N, 3]
                'velocity': velocity, # [T, N, 3]
                'principal_stress': principal_stress,# [T, N, 3]
                'principal_strain': principal_strain,# [T, N, 3]
                'stress_components': stress_components, # [T, N, 6]
                'strain_components': strain_components, # [T, N, 6]
            }
        }
       
        # Save to SQLite
        buf = io.BytesIO()
        torch.save(temporal_solution, buf)
        buf.seek(0)
       
        conn = sqlite3.connect(SCRIPT_DIR / "laser_temporal.db", check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temporal_sims (
                name TEXT PRIMARY KEY,
                P_W REAL, V_mm_s REAL, n_timesteps INTEGER, n_points INTEGER,
                total_time REAL, pt_data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
       
        conn.execute("""
            INSERT OR REPLACE INTO temporal_sims
            (name, P_W, V_mm_s, n_timesteps, n_points, total_time, pt_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sim["name"], sim["P"], sim["V"], n_timesteps, n_points, sim["total_time"], buf.read()))
        conn.commit()
        conn.close()
       
        st.success(f"**{sim['name']}** → Temporal .pt sequence saved!")
        st.balloons()
       
        # Download button
        st.download_button(
            "📥 Download Temporal .pt",
            buf.getvalue(),
            f"{sim['name']}_temporal_sequence.pt",
            help="Contains full temporal sequence with proper time stepping"
        )
# --------------------------------------------------------------
# Interpolate Across P and V
# --------------------------------------------------------------
st.subheader("Interpolate Across P and V")

# Load all saved temporal sims
@st.cache_resource
def get_db():
    conn = sqlite3.connect(SCRIPT_DIR / "laser_temporal.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS temporal_sims (
            name TEXT PRIMARY KEY,
            P_W REAL, V_mm_s REAL, n_timesteps INTEGER, n_points INTEGER,
            total_time REAL, pt_data BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn

conn = get_db()
rows = conn.execute("SELECT name, P_W, V_mm_s, pt_data FROM temporal_sims").fetchall()
sims_data = {row[0]: {'metadata': {'P_W': row[1], 'V_mm_s': row[2]}, 'temporal_data': torch.load(io.BytesIO(row[3]))['temporal_data']} for row in rows}

if len(sims_data) < 2:
    st.warning("Need at least 2 simulations for interpolation.")
else:
    # Extract params (P, V)
    params_list = [(data['metadata']['P_W'], data['metadata']['V_mm_s']) for data in sims_data.values()]

    # Target params
    target_P = st.number_input("Target P (W)", min_value=min(p[0] for p in params_list), max_value=max(p[0] for p in params_list), value=(min(p[0] for p in params_list) + max(p[0] for p in params_list)) / 2)
    target_V = st.number_input("Target V (mm/s)", min_value=min(p[1] for p in params_list), max_value=max(p[1] for p in params_list), value=(min(p[1] for p in params_list) + max(p[1] for p in params_list)) / 2)

    if st.button("Interpolate Solution"):
        class LaserAttentionInterpolator(nn.Module):
            def __init__(self, sigma=0.2, num_heads=4, d_head=8):
                super().__init__()
                self.sigma = sigma
                self.num_heads = num_heads
                self.d_head = d_head
                self.W_q = nn.Linear(2, self.num_heads * self.d_head)  # P, V
                self.W_k = nn.Linear(2, self.num_heads * self.d_head)

            def forward(self, sims_data, params_list, target_P, target_V):
                # Normalize params
                Ps = np.array([p[0] for p in params_list])
                Vs = np.array([p[1] for p in params_list])
                min_P, max_P = min(Ps), max(Ps)
                min_V, max_V = min(Vs), max(Vs)
                P_norm = (Ps - min_P) / (max_P - min_P + 1e-6)
                V_norm = (Vs - min_V) / (max_V - min_V + 1e-6)
                target_P_norm = (target_P - min_P) / (max_P - min_P + 1e-6)
                target_V_norm = (target_V - min_V) / (max_V - min_V + 1e-6)

                params_tensor = torch.tensor(np.stack([P_norm, V_norm], axis=1), dtype=torch.float32)  # [N, 2]
                target_params = torch.tensor([[target_P_norm, target_V_norm]], dtype=torch.float32)  # [1, 2]

                queries = self.W_q(target_params).view(1, self.num_heads, self.d_head)
                keys = self.W_k(params_tensor).view(len(params_list), self.num_heads, self.d_head)
                attn_logits = torch.einsum('bhd,mhd->bmh', queries, keys) / np.sqrt(self.d_head)
                attn_weights = torch.softmax(attn_logits, dim=1).mean(dim=2).squeeze(0)  # [N]

                # Gaussian spatial weights
                dists = torch.sqrt(
                    ((torch.tensor(P_norm) - target_P_norm) / self.sigma)**2 +
                    ((torch.tensor(V_norm) - target_V_norm) / self.sigma)**2
                )
                spatial_weights = torch.exp(-dists**2 / 2)
                spatial_weights /= spatial_weights.sum() + 1e-6

                combined_weights = attn_weights * spatial_weights
                combined_weights /= combined_weights.sum() + 1e-6

                return self._interpolate_fields(sims_data, combined_weights, target_P, target_V)

            def _interpolate_fields(self, sims_data, weights, target_P, target_V):
                # Find common time grid (e.g., union of all times, sorted unique)
                all_times = np.unique(np.concatenate([data['temporal_data']['times'].numpy() for data in sims_data.values()]))
                common_times = torch.tensor(all_times, dtype=torch.float32)

                # Interpolate each sim to common times
                interp_sims = {}
                for name, data in sims_data.items():
                    orig_times = data['temporal_data']['times'].numpy()
                    interp_data = {}
                    for field, tensor in data['temporal_data'].items():
                        if field == 'times':
                            continue
                        # Interp along time dim (axis=0)
                        if tensor.ndim == 2:  # [T, N]
                            f = interp1d(orig_times, tensor.numpy(), axis=0, kind='linear', fill_value='extrapolate')
                            interp_data[field] = torch.tensor(f(common_times.numpy()))
                        elif tensor.ndim == 3:  # [T, N, D]
                            f = interp1d(orig_times, tensor.numpy(), axis=0, kind='linear', fill_value='extrapolate')
                            interp_data[field] = torch.tensor(f(common_times.numpy()))
                    interp_sims[name] = interp_data

                # Weighted sum across sims
                first_interp = next(iter(interp_sims.values()))
                interp_temporal_data = {k: torch.zeros_like(v) for k, v in first_interp.items()}
                for i, (name, i_data) in enumerate(interp_sims.items()):
                    w = weights[i]
                    for field in interp_temporal_data:
                        interp_temporal_data[field] += w * i_data[field]

                # Metadata
                metadata = next(iter(sims_data.values()))['metadata'].copy()
                metadata.update({
                    'sim_name': f"Interp_P{target_P}_V{target_V}",
                    'P_W': target_P,
                    'V_mm_s': target_V,
                    'n_timesteps': len(common_times),
                    'total_time': common_times[-1].item(),
                    'interpolated': True,
                    'attention_weights': weights.tolist()
                })

                interp_solution = {
                    'metadata': metadata,
                    'temporal_data': {'times': common_times, **interp_temporal_data}
                }

                # Save interpolated to DB
                buf = io.BytesIO()
                torch.save(interp_solution, buf)
                buf.seek(0)

                conn = get_db()
                conn.execute("""
                    INSERT OR REPLACE INTO temporal_sims
                    (name, P_W, V_mm_s, n_timesteps, n_points, total_time, pt_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (metadata['sim_name'], target_P, target_V, len(common_times), metadata['n_points'], metadata['total_time'], buf.read()))
                conn.commit()

                return interp_solution

        with st.spinner("Interpolating..."):
            interpolator = LaserAttentionInterpolator()
            interp_solution = interpolator.forward(sims_data, params_list, target_P, target_V)
            st.success(f"Interpolated for P={target_P}W, V={target_V}mm/s!")
            buf = io.BytesIO()
            torch.save(interp_solution, buf)
            buf.seek(0)
            st.download_button("📥 Download Interpolated .pt", buf.getvalue(), f"Interp_P{target_P}_V{target_V}_temporal.pt")
# --------------------------------------------------------------
# Temporal Visualization
# --------------------------------------------------------------
st.subheader("🕐 Temporal Visualization")
# Load and visualize temporal data

# Initialise slider key in session state
if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0

# Load list of saved simulations (cached)
@st.cache_data
def list_saved_sims():
    conn = get_db()
    return [row[0] for row in conn.execute("SELECT name FROM temporal_sims").fetchall()]

saved = list_saved_sims()

if saved:
    load_name = st.selectbox("Select saved temporal sim", saved)
   
    if st.button("Load Temporal Sequence"):
        # ---- cached heavy load ---------------------------------
        @st.cache_data
        def load_pt(_name):
            conn = get_db()
            row = conn.execute(
                "SELECT pt_data FROM temporal_sims WHERE name = ?", (_name,)
            ).fetchone()
            return torch.load(io.BytesIO(row[0])) if row else None

        data = load_pt(load_name)
        # ---------------------------------------------------------

        if data:
            st.session_state.loaded_data = data   # keep in session for slider
            st.session_state.t_idx = 0           # reset slider

    # ---- render only when data are in session -----------------
    if "loaded_data" in st.session_state:
        data = st.session_state.loaded_data

        st.write(f"**Loaded:** {load_name}")
        st.write(f"**Timesteps:** {data['metadata']['n_timesteps']}")
        st.write(f"**Total time:** {data['metadata']['total_time']:.6f} s")

        # Slider that survives reruns
        t_idx = st.slider(
            "Time step",
            0,
            data['metadata']['n_timesteps'] - 1,
            st.session_state.t_idx,
            key="t_idx"          # binds to session_state automatically
        )
        current_time = data['temporal_data']['times'][t_idx].item()
        st.write(f"**Current time:** {current_time:.6f} s")
           
        # Field selection
        field = st.selectbox("Field to visualize",
                             ["temperature", "vonmises_stress", "pressure",
                              "displacement_mag", "velocity_mag"])
           
        # Get data for current timestep
        pts = data['temporal_data']['coordinates'][t_idx].numpy()
           
        if field == "temperature":
            vals = data['temporal_data']['temperature'][t_idx].numpy()
        elif field == "vonmises_stress":
            vals = data['temporal_data']['vonmises_stress'][t_idx].numpy()
        elif field == "pressure":
            vals = data['temporal_data']['pressure'][t_idx].numpy()
        elif field == "displacement_mag":
            disp = data['temporal_data']['displacement'][t_idx].numpy()
            vals = np.linalg.norm(disp, axis=1)
        else: # velocity_mag
            vel = data['temporal_data']['velocity'][t_idx].numpy()
            vals = np.linalg.norm(vel, axis=1)
           
        # Plot
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(
                size=2,
                color=vals,
                colorscale='Hot',
                opacity=0.8,
                colorbar=dict(title=field)
            )
        ))
        fig.update_layout(
            scene_aspectmode='data',
            title=f"{field} at t = {current_time:.6f} s",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.markdown("**Ready for cross-attention interpolation between P-V parameters!**")
