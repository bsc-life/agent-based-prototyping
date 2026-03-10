"""
Streamlit GUI for Diffusion Benchmark Configuration and Execution.

Run with:
    streamlit run examples/gui_benchmark.py

This provides a visual interface for:
- Configuring simulation parameters (domain, grid, time stepping)
- Setting physical parameters (diffusion, decay)
- Choosing initial and boundary conditions
- Adding agents and bulk regions
- Selecting numerical methods to benchmark
- Configuring golden solution (analytical or numerical reference)
- Running benchmarks and viewing results
"""

import streamlit as st
import numpy as np
import sys
import os
import io
import time
from pathlib import Path
from contextlib import redirect_stdout

# streamlit run gui_benchmark.py

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarking import (
    BenchmarkRunner,
    create_scenario_with_numerical_reference,
    create_scenario,
    get_scenario_by_name,
)
from diffusion_schemas import (
    ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema,
    ImplicitLODSchema, CrankNicolsonLODSchema,
    ExplicitEulerBCSchema, ImplicitEulerBCSchema, CrankNicolsonBCSchema,
    ImplicitLODBCSchema, CrankNicolsonLODBCSchema
)
from diffusion_schemas.utils import gaussian, uniform, DirichletBC, NeumannBC

# =============================================================================
# Constants
# =============================================================================

SCHEMA_MAP = {
    "Explicit Euler": ExplicitEulerSchema,
    "Implicit Euler": ImplicitEulerSchema,
    "Crank-Nicolson": CrankNicolsonSchema,
    "Implicit LOD": ImplicitLODSchema,
    "Crank-Nicolson LOD": CrankNicolsonLODSchema,
}

SCHEMA_BC_MAP = {
    "Explicit Euler (BC)": ExplicitEulerBCSchema,
    "Implicit Euler (BC)": ImplicitEulerBCSchema,
    "Crank-Nicolson (BC)": CrankNicolsonBCSchema,
    "Implicit LOD (BC)": ImplicitLODBCSchema,
    "Crank-Nicolson LOD (BC)": CrankNicolsonLODBCSchema,
}

ALL_SCHEMAS = {**SCHEMA_MAP, **SCHEMA_BC_MAP}

GOLDEN_SOLUTION_TYPES = [
    "Numerical Reference",
    "Analytical: Gaussian 1D",
    "Analytical: Gaussian 2D",
    "Analytical: Gaussian 3D",
    "Analytical: Step Function 1D",
    "Analytical: Exponential Decay",
    "Analytical: Sine Decay 1D",
]

PREBUILT_SCENARIOS = [
    "-- Custom --",
    "gaussian_pulse_1d",
    "gaussian_pulse_2d",
    "gaussian_pulse_3d",
    "step_function_1d",
    "step_function_2d",
    "steady_state_agent_1d",
    "steady_state_agent_2d",
    "exponential_decay_1d",
    "sine_decay_1d",
    "cosine_diffusion_1d",
    "single_tumor_2d",
]

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Diffusion Benchmark GUI",
    page_icon="🔬",
    layout="wide",
)

st.title("Diffusion Schema Benchmark Interface")
st.markdown("Configure and run diffusion benchmarks visually.")

# =============================================================================
# Sidebar: prebuilt scenario loader
# =============================================================================

with st.sidebar:
    st.header("Quick Load")
    prebuilt = st.selectbox("Load a prebuilt scenario", PREBUILT_SCENARIOS)
    if prebuilt != "-- Custom --":
        if st.button("Load scenario"):
            try:
                sc = get_scenario_by_name(prebuilt)
                st.session_state["loaded_scenario"] = sc
                st.success(f"Loaded: {prebuilt}")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.header("Output")
    output_dir = st.text_input("Output directory", value="benchmark_results/gui_run")
    store_history = st.checkbox("Store time history", value=True)
    generate_plots = st.checkbox("Generate plots", value=True)

# =============================================================================
# Helper: get value from loaded scenario or default
# =============================================================================

def _get(key, default=None):
    """Get a value from a loaded scenario or return a default."""
    sc = st.session_state.get("loaded_scenario")
    if sc and key in sc:
        return sc[key]
    return default

# =============================================================================
# Tabs
# =============================================================================

tab_sim, tab_physics, tab_ic, tab_bc, tab_agents, tab_bulk, tab_golden, tab_methods, tab_run = st.tabs([
    "Simulation", "Physics", "Initial Condition", "Boundary Conditions",
    "Agents", "Bulk Regions", "Golden Solution", "Methods", "Run"
])

# ---- TAB: Simulation --------------------------------------------------------

with tab_sim:
    st.subheader("Domain & Discretisation")

    ndim = st.radio("Dimensionality", [1, 2, 3], horizontal=True,
                    index=0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Domain size**")
        if ndim == 1:
            default_ds = _get("domain_size", 1.0)
            if isinstance(default_ds, (list, tuple)):
                default_ds = default_ds[0]
            domain_size_x = st.number_input("L_x", value=float(default_ds), format="%.4f", key="ds_x")
            domain_size = domain_size_x
        elif ndim == 2:
            default_ds = _get("domain_size", (1.0, 1.0))
            if not isinstance(default_ds, (list, tuple)):
                default_ds = (default_ds, default_ds)
            domain_size_x = st.number_input("L_x", value=float(default_ds[0]), format="%.4f", key="ds_x")
            domain_size_y = st.number_input("L_y", value=float(default_ds[1]), format="%.4f", key="ds_y")
            domain_size = (domain_size_x, domain_size_y)
        else:
            default_ds = _get("domain_size", (1.0, 1.0, 1.0))
            if not isinstance(default_ds, (list, tuple)):
                default_ds = (default_ds, default_ds, default_ds)
            domain_size_x = st.number_input("L_x", value=float(default_ds[0]), format="%.4f", key="ds_x")
            domain_size_y = st.number_input("L_y", value=float(default_ds[1]), format="%.4f", key="ds_y")
            domain_size_z = st.number_input("L_z", value=float(default_ds[2]), format="%.4f", key="ds_z")
            domain_size = (domain_size_x, domain_size_y, domain_size_z)

    with col2:
        st.markdown("**Grid points**")
        if ndim == 1:
            default_gp = _get("grid_points", 100)
            if isinstance(default_gp, (list, tuple)):
                default_gp = default_gp[0]
            gp_x = st.number_input("N_x", value=int(default_gp), step=1, key="gp_x")
            grid_points = gp_x
        elif ndim == 2:
            default_gp = _get("grid_points", (50, 50))
            if not isinstance(default_gp, (list, tuple)):
                default_gp = (default_gp, default_gp)
            gp_x = st.number_input("N_x", value=int(default_gp[0]), step=1, key="gp_x")
            gp_y = st.number_input("N_y", value=int(default_gp[1]), step=1, key="gp_y")
            grid_points = (gp_x, gp_y)
        else:
            default_gp = _get("grid_points", (30, 30, 30))
            if not isinstance(default_gp, (list, tuple)):
                default_gp = (default_gp, default_gp, default_gp)
            gp_x = st.number_input("N_x", value=int(default_gp[0]), step=1, key="gp_x")
            gp_y = st.number_input("N_y", value=int(default_gp[1]), step=1, key="gp_y")
            gp_z = st.number_input("N_z", value=int(default_gp[2]), step=1, key="gp_z")
            grid_points = (gp_x, gp_y, gp_z)

    st.divider()
    st.markdown("**Time stepping**")
    col_dt, col_tf = st.columns(2)
    with col_dt:
        dt = st.number_input("dt (time step)", value=float(_get("dt", 0.001)),
                             format="%.6f", min_value=1e-10)
    with col_tf:
        t_final = st.number_input("t_final", value=float(_get("t_final", 0.5)),
                                  format="%.4f", min_value=0.0)

    # Show computed dx
    if ndim == 1:
        dx_val = domain_size / max(grid_points - 1, 1)
        st.info(f"Effective dx = {dx_val:.6f}")
    else:
        if isinstance(domain_size, (list, tuple)):
            gp_tuple = grid_points if isinstance(grid_points, (list, tuple)) else (grid_points,)
            dx_vals = [d / max(n - 1, 1) for d, n in zip(domain_size, gp_tuple)]
            st.info(f"Effective dx = {', '.join(f'{v:.6f}' for v in dx_vals)}")

# ---- TAB: Physics ------------------------------------------------------------

with tab_physics:
    st.subheader("Physical Parameters")

    diffusion_coefficient = st.number_input(
        "Diffusion coefficient (D)",
        value=float(_get("diffusion_coefficient", 0.01)),
        format="%.6e", min_value=0.0
    )
    decay_rate = st.number_input(
        "Decay rate (λ)",
        value=float(_get("decay_rate", 0.0)),
        format="%.6e", min_value=0.0
    )

# ---- TAB: Initial Condition --------------------------------------------------

with tab_ic:
    st.subheader("Initial Condition")

    ic_types = ["uniform", "gaussian", "step_function", "checkerboard", "sphere", "sine"]
    default_ic = _get("initial_condition", {"type": "uniform", "value": 0.0})
    default_ic_type = default_ic.get("type", "uniform") if isinstance(default_ic, dict) else "uniform"
    ic_type = st.selectbox("Type", ic_types, index=ic_types.index(default_ic_type) if default_ic_type in ic_types else 0)

    ic_spec = {"type": ic_type}

    if ic_type == "uniform":
        ic_spec["value"] = st.number_input("Value", value=float(default_ic.get("value", 0.0) if isinstance(default_ic, dict) else 0.0), format="%.4f")

    elif ic_type == "gaussian":
        _def = default_ic if isinstance(default_ic, dict) and default_ic.get("type") == "gaussian" else {}
        if ndim == 1:
            ic_spec["center"] = st.number_input("Center", value=float(_def.get("center", 0.5)))
        elif ndim == 2:
            c = _def.get("center", (0.5, 0.5))
            c = c if isinstance(c, (list, tuple)) else (c, c)
            cx = st.number_input("Center X", value=float(c[0]))
            cy = st.number_input("Center Y", value=float(c[1]))
            ic_spec["center"] = (cx, cy)
        else:
            c = _def.get("center", (0.5, 0.5, 0.5))
            c = c if isinstance(c, (list, tuple)) else (c, c, c)
            cx = st.number_input("Center X", value=float(c[0]))
            cy = st.number_input("Center Y", value=float(c[1]))
            cz = st.number_input("Center Z", value=float(c[2]))
            ic_spec["center"] = (cx, cy, cz)
        ic_spec["amplitude"] = st.number_input("Amplitude", value=float(_def.get("amplitude", 1.0)))
        ic_spec["width"] = st.number_input("Width (σ)", value=float(_def.get("width", 0.1)), format="%.4f")

    elif ic_type == "step_function":
        _def = default_ic if isinstance(default_ic, dict) and default_ic.get("type") == "step_function" else {}
        ic_spec["position"] = st.number_input("Step position", value=float(_def.get("position", 0.5)))
        ic_spec["value_left"] = st.number_input("Value left", value=float(_def.get("value_left", 1.0)))
        ic_spec["value_right"] = st.number_input("Value right", value=float(_def.get("value_right", 0.0)))
        ic_spec["axis"] = st.number_input("Axis", value=int(_def.get("axis", 0)), min_value=0, max_value=max(ndim - 1, 0))

    elif ic_type == "checkerboard":
        _def = default_ic if isinstance(default_ic, dict) and default_ic.get("type") == "checkerboard" else {}
        ic_spec["spacing"] = st.number_input("Spacing", value=float(_def.get("spacing", 1.0)))
        ic_spec["value_on"] = st.number_input("Value on", value=float(_def.get("value_on", 1.0)))
        ic_spec["value_off"] = st.number_input("Value off", value=float(_def.get("value_off", 0.0)))

    elif ic_type == "sphere":
        _def = default_ic if isinstance(default_ic, dict) and default_ic.get("type") == "sphere" else {}
        if ndim == 1:
            ic_spec["center"] = st.number_input("Center", value=0.5)
        elif ndim == 2:
            c = _def.get("center", (0.5, 0.5))
            c = c if isinstance(c, (list, tuple)) else (c, c)
            ic_spec["center"] = (
                st.number_input("Center X", value=float(c[0]), key="ic_sph_cx"),
                st.number_input("Center Y", value=float(c[1]), key="ic_sph_cy"),
            )
        else:
            c = _def.get("center", (0.5, 0.5, 0.5))
            c = c if isinstance(c, (list, tuple)) else (c, c, c)
            ic_spec["center"] = (
                st.number_input("Center X", value=float(c[0]), key="ic_sph_cx"),
                st.number_input("Center Y", value=float(c[1]), key="ic_sph_cy"),
                st.number_input("Center Z", value=float(c[2]), key="ic_sph_cz"),
            )
        ic_spec["radius"] = st.number_input("Radius", value=float(_def.get("radius", 0.2)))
        ic_spec["value_inside"] = st.number_input("Value inside", value=float(_def.get("value_inside", 1.0)))
        ic_spec["value_outside"] = st.number_input("Value outside", value=float(_def.get("value_outside", 0.0)))

    elif ic_type == "sine":
        _def = default_ic if isinstance(default_ic, dict) and default_ic.get("type") == "sine" else {}
        ic_spec["wavenumber"] = st.number_input("Wavenumber", value=float(_def.get("wavenumber", 1.0)))
        ic_spec["amplitude"] = st.number_input("Amplitude", value=float(_def.get("amplitude", 1.0)), key="ic_sine_amp")

# ---- TAB: Boundary Condition -------------------------------------------------

with tab_bc:
    st.subheader("Boundary Condition")

    # bc_types = ["neumann", "dirichlet", "periodic", "robin"]
    bc_types = ["neumann", "dirichlet"]  # for now, only support these in the GUI
    default_bc = _get("boundary_condition", {"type": "neumann", "flux": 0.0})
    default_bc_type = default_bc.get("type", "neumann") if isinstance(default_bc, dict) else "neumann"
    bc_type = st.selectbox("Type", bc_types, index=bc_types.index(default_bc_type) if default_bc_type in bc_types else 0)

    bc_spec = {"type": bc_type}

    if bc_type == "dirichlet":
        _def = default_bc if isinstance(default_bc, dict) and default_bc.get("type") == "dirichlet" else {}
        raw = _def.get("values", _def.get("value", 0.0))
        st.markdown("Enter a single value (applied to all faces) or comma‑separated values for each face.")
        val_str = st.text_input("Dirichlet value(s)", value=str(raw))
        try:
            parts = [float(v.strip()) for v in val_str.split(",")]
            bc_spec["values"] = parts[0] if len(parts) == 1 else parts
        except ValueError:
            bc_spec["values"] = 0.0
            st.warning("Invalid input — defaulting to 0.0")

    elif bc_type == "neumann":
        _def = default_bc if isinstance(default_bc, dict) and default_bc.get("type") == "neumann" else {}
        bc_spec["flux"] = st.number_input("Flux", value=float(_def.get("flux", _def.get("values", 0.0))), format="%.6f")

    elif bc_type == "periodic":
        st.info("Periodic BCs have no additional parameters.")

    elif bc_type == "robin":
        _def = default_bc if isinstance(default_bc, dict) and default_bc.get("type") == "robin" else {}
        bc_spec["alpha"] = st.number_input("alpha", value=float(_def.get("alpha", 1.0)))
        bc_spec["beta"] = st.number_input("beta", value=float(_def.get("beta", 1.0)))
        bc_spec["gamma"] = st.number_input("gamma", value=float(_def.get("gamma", 0.0)))

# ---- TAB: Agents -------------------------------------------------------------

with tab_agents:
    st.subheader("Point-Source Agents")
    st.markdown("Add agents that act as localised sources/sinks.")

    if "agents_list" not in st.session_state:
        default_agents = _get("agents", None)
        st.session_state["agents_list"] = list(default_agents) if default_agents else []

    num_agents = st.number_input("Number of agents", min_value=0, value=len(st.session_state["agents_list"]), step=1)

    # Resize list
    while len(st.session_state["agents_list"]) < num_agents:
        st.session_state["agents_list"].append({
            "position": [0.5] * ndim,
            "net_rate": 1.0,
        })
    st.session_state["agents_list"] = st.session_state["agents_list"][:num_agents]

    agents_list = st.session_state["agents_list"]

    for i in range(num_agents):
        with st.expander(f"Agent {i + 1}", expanded=(i == 0)):
            agent = agents_list[i]

            # Agent type
            is_complete = st.checkbox("Use CompleteAgent (secretion/uptake/saturation)",
                                      value="saturation_density" in agent,
                                      key=f"agent_complete_{i}")

            # Position
            pos = agent.get("position", [0.5] * ndim)
            if not isinstance(pos, (list, tuple)):
                pos = [pos]
            pos = list(pos)
            while len(pos) < ndim:
                pos.append(0.5)

            new_pos = []
            cols = st.columns(ndim)
            for d in range(ndim):
                with cols[d]:
                    v = st.number_input(f"pos[{d}]", value=float(pos[d]), format="%.4f", key=f"agent_pos_{i}_{d}")
                    new_pos.append(v)
            agent["position"] = tuple(new_pos) if ndim > 1 else (new_pos[0],)

            if is_complete:
                agent["secretion_rate"] = st.number_input("Secretion rate", value=float(agent.get("secretion_rate", 1.0)), key=f"agent_sec_{i}")
                agent["uptake_rate"] = st.number_input("Uptake rate", value=float(agent.get("uptake_rate", 0.0)), key=f"agent_upt_{i}")
                agent["saturation_density"] = st.number_input("Saturation density", value=float(agent.get("saturation_density", 0.0)), key=f"agent_sat_{i}")
                agent.pop("net_rate", None)
            else:
                agent["net_rate"] = st.number_input("Net rate", value=float(agent.get("net_rate", 1.0)), key=f"agent_nr_{i}")
                agent.pop("secretion_rate", None)
                agent.pop("uptake_rate", None)
                agent.pop("saturation_density", None)

            kw = st.number_input("Kernel width (0 = point source)", value=float(agent.get("kernel_width", 0.0) or 0.0),
                                 min_value=0.0, format="%.4f", key=f"agent_kw_{i}")
            agent["kernel_width"] = kw if kw > 0 else None

            agent["name"] = st.text_input("Name (optional)", value=agent.get("name", ""), key=f"agent_name_{i}")

    agents_spec = agents_list if num_agents > 0 else None

# ---- TAB: Bulk Regions -------------------------------------------------------

with tab_bulk:
    st.subheader("Bulk Source/Sink Regions")
    st.markdown("Define volumetric regions with a net secretion/uptake rate.")

    if "bulk_regions" not in st.session_state:
        default_bulk = _get("bulk", None)
        if default_bulk and isinstance(default_bulk, dict):
            st.session_state["bulk_regions"] = list(default_bulk.get("regions", []))
        else:
            st.session_state["bulk_regions"] = []

    num_regions = st.number_input("Number of bulk regions", min_value=0,
                                  value=len(st.session_state["bulk_regions"]), step=1)

    while len(st.session_state["bulk_regions"]) < num_regions:
        st.session_state["bulk_regions"].append({
            "type": "sphere",
            "center": tuple([0.5] * ndim),
            "radius": 0.2,
            "net_rate": 0.0,
            "name": "",
        })
    st.session_state["bulk_regions"] = st.session_state["bulk_regions"][:num_regions]

    regions_list = st.session_state["bulk_regions"]

    for i in range(num_regions):
        with st.expander(f"Region {i + 1}", expanded=(i == 0)):
            region = regions_list[i]
            rtype = st.selectbox("Shape", ["sphere", "rectangle"], key=f"reg_type_{i}",
                                 index=0 if region.get("type") == "sphere" else 1)
            region["type"] = rtype

            if rtype == "sphere":
                center = region.get("center", tuple([0.5] * ndim))
                if not isinstance(center, (list, tuple)):
                    center = (center,)
                center = list(center)
                while len(center) < ndim:
                    center.append(0.5)
                new_center = []
                cols = st.columns(ndim)
                for d in range(ndim):
                    with cols[d]:
                        v = st.number_input(f"center[{d}]", value=float(center[d]), format="%.4f", key=f"reg_c_{i}_{d}")
                        new_center.append(v)
                region["center"] = tuple(new_center)
                region["radius"] = st.number_input("Radius", value=float(region.get("radius", 0.2)),
                                                   format="%.4f", key=f"reg_r_{i}")
                region.pop("origin", None)
                region.pop("size", None)

            else:  # rectangle
                origin = region.get("origin", tuple([0.0] * ndim))
                if not isinstance(origin, (list, tuple)):
                    origin = (origin,)
                origin = list(origin)
                while len(origin) < ndim:
                    origin.append(0.0)
                new_origin = []
                cols = st.columns(ndim)
                for d in range(ndim):
                    with cols[d]:
                        v = st.number_input(f"origin[{d}]", value=float(origin[d]), format="%.4f", key=f"reg_o_{i}_{d}")
                        new_origin.append(v)
                region["origin"] = tuple(new_origin)

                sz = region.get("size", tuple([0.2] * ndim))
                if not isinstance(sz, (list, tuple)):
                    sz = (sz,)
                sz = list(sz)
                while len(sz) < ndim:
                    sz.append(0.2)
                new_size = []
                cols2 = st.columns(ndim)
                for d in range(ndim):
                    with cols2[d]:
                        v = st.number_input(f"size[{d}]", value=float(sz[d]), format="%.4f", key=f"reg_s_{i}_{d}")
                        new_size.append(v)
                region["size"] = tuple(new_size)
                region.pop("center", None)
                region.pop("radius", None)

            region["net_rate"] = st.number_input("Net rate", value=float(region.get("net_rate", 0.0)),
                                                 format="%.4f", key=f"reg_nr_{i}")
            region["name"] = st.text_input("Name (optional)", value=region.get("name", ""), key=f"reg_name_{i}")

    bulk_spec = {"regions": regions_list} if num_regions > 0 else None

# ---- TAB: Golden Solution ----------------------------------------------------

with tab_golden:
    st.subheader("Golden (Reference) Solution")
    st.markdown("Choose how the benchmark validates results.")

    golden_type = st.selectbox("Golden solution type", GOLDEN_SOLUTION_TYPES)

    golden_spec = None  # default: no validation

    if golden_type == "None (no validation)":
        st.info("No golden solution — benchmark will only report timing, no error metrics.")

    elif golden_type.startswith("Analytical: Gaussian"):
        dim_label = golden_type.split()[-1]  # "1D", "2D", "3D"
        dim_n = int(dim_label[0])
        gs_type = f"gaussian_{dim_label.lower()}"

        if dim_n == 1:
            gs_center = st.number_input("Center", value=0.5, key="gs_g_c")
        elif dim_n == 2:
            c1 = st.number_input("Center X", value=0.5, key="gs_g_cx")
            c2 = st.number_input("Center Y", value=0.5, key="gs_g_cy")
            gs_center = (c1, c2)
        else:
            c1 = st.number_input("Center X", value=0.5, key="gs_g_cx")
            c2 = st.number_input("Center Y", value=0.5, key="gs_g_cy")
            c3 = st.number_input("Center Z", value=0.5, key="gs_g_cz")
            gs_center = (c1, c2, c3)

        gs_amp = st.number_input("Amplitude", value=1.0, key="gs_g_a")
        gs_w = st.number_input("Initial width", value=0.1, format="%.4f", key="gs_g_w")
        gs_D = st.number_input("Diffusion coeff (for analytical)", value=float(diffusion_coefficient), format="%.6e", key="gs_g_D")

        golden_spec = {
            "type": gs_type,
            "center": gs_center,
            "amplitude": gs_amp,
            "initial_width": gs_w,
            "diffusion_coefficient": gs_D,
        }

    elif golden_type == "Analytical: Step Function 1D":
        golden_spec = {
            "type": "step_function_1d",
            "domain_length": float(domain_size) if isinstance(domain_size, (int, float)) else float(domain_size[0]),
            "position": st.number_input("Step position", value=0.5, key="gs_sf_pos"),
            "value_left": st.number_input("Value left", value=1.0, key="gs_sf_vl"),
            "value_right": st.number_input("Value right", value=0.0, key="gs_sf_vr"),
            "axis": st.number_input("Axis", value=0, min_value=0, key="gs_sf_ax"),
            "diffusion_coefficient": st.number_input("D (analytical)", value=float(diffusion_coefficient), format="%.6e", key="gs_sf_D"),
            "n_terms": st.number_input("Fourier terms", value=200, step=10, key="gs_sf_n"),
        }

    elif golden_type == "Analytical: Exponential Decay":
        st.info("Uses the IC you configured above with the specified decay rate.")
        golden_spec = {
            "type": "exponential_decay",
            "decay_rate": float(decay_rate),
        }

    elif golden_type == "Analytical: Sine Decay 1D":
        golden_spec = {
            "type": "sine_decay_1d",
            "wavenumber": st.number_input("Wavenumber", value=1.0, key="gs_sd_wn"),
            "amplitude": st.number_input("Amplitude", value=1.0, key="gs_sd_amp"),
            "diffusion_coefficient": st.number_input("D", value=float(diffusion_coefficient), format="%.6e", key="gs_sd_D"),
        }

    elif golden_type == "Numerical Reference":
        st.markdown("A high‑resolution simulation is run first and used as reference.")
        ref_schema_name = st.selectbox("Reference schema class", list(ALL_SCHEMAS.keys()),
                                       index=list(ALL_SCHEMAS.keys()).index("Implicit LOD (BC)"))
        ref_schema_class = ALL_SCHEMAS[ref_schema_name]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            dx_ref = st.number_input("dx_ref (reference spatial resolution)", value=0.001, format="%.6f", key="gs_nr_dx")
        with col_r2:
            dt_ref = st.number_input("dt_ref (reference time step)", value=0.001, format="%.6f", key="gs_nr_dt")

        golden_spec = {
            "type": "numerical_reference",
            "schema_class": ref_schema_class,
            "dx_ref": dx_ref,
            "dt_ref": dt_ref,
        }

# ---- TAB: Methods ------------------------------------------------------------

with tab_methods:
    st.subheader("Numerical Methods to Benchmark")
    st.markdown("Select which schemas to run against the scenario.")

    col_no_bc, col_bc = st.columns(2)

    with col_no_bc:
        st.markdown("**BC applied through Operator Splitting**")
        selected_no_bc = []
        for name in SCHEMA_MAP:
            if st.checkbox(name, value=False, key=f"method_{name}"):
                selected_no_bc.append(name)

    with col_bc:
        st.markdown("**BC integrated into schema equations**")
        selected_bc = []
        for name in SCHEMA_BC_MAP:
            if st.checkbox(name, value=True, key=f"method_{name}"):
                selected_bc.append(name)

    selected_methods = selected_no_bc + selected_bc

    if not selected_methods:
        st.warning("Select at least one method to run a benchmark.")

# ---- TAB: Run ----------------------------------------------------------------

with tab_run:
    st.subheader("Run Benchmark")

    # Build scenario name
    scenario_name = st.text_input("Scenario name", value=_get("name", "gui_scenario"))
    scenario_desc = st.text_input("Description (optional)", value=_get("description", ""))

    st.divider()

    # Summary
    with st.expander("Scenario summary", expanded=True):
        st.json({
            "name": scenario_name,
            "domain_size": domain_size if isinstance(domain_size, (int, float)) else list(domain_size),
            "grid_points": grid_points if isinstance(grid_points, (int, float)) else list(grid_points),
            "dt": dt,
            "t_final": t_final,
            "diffusion_coefficient": diffusion_coefficient,
            "decay_rate": decay_rate,
            "initial_condition": ic_spec,
            "boundary_condition": bc_spec,
            "agents": agents_spec,
            "bulk": bulk_spec,
            "golden_solution": golden_type,
            "methods": selected_methods,
        })

    st.divider()

    if st.button("Run", type="primary", disabled=(len(selected_methods) == 0)):
        # ---- Build scenario --------------------------------------------------
        use_num_ref = (golden_type == "Numerical Reference")

        if use_num_ref:
            scenario = create_scenario_with_numerical_reference(
                name=scenario_name,
                schema_class=golden_spec.get("schema_class"),
                domain_size=domain_size,
                grid_points=grid_points,
                dt=dt,
                t_final=t_final,
                initial_condition=ic_spec,
                diffusion_coefficient=diffusion_coefficient,
                decay_rate=decay_rate,
                boundary_condition=bc_spec,
                agents=agents_spec,
                bulk=bulk_spec,
                dx_ref=golden_spec["dx_ref"],
                dt_ref=golden_spec["dt_ref"],
                description=scenario_desc,
            )
        else:
            # Exponential decay needs IC callable patched in
            if golden_spec and golden_spec.get("type") == "exponential_decay":
                from benchmarking.scenarios import _build_initial_condition
                golden_spec["initial_condition"] = _build_initial_condition(ic_spec)

            scenario = create_scenario(
                name=scenario_name,
                domain_size=domain_size,
                grid_points=grid_points,
                dt=dt,
                t_final=t_final,
                initial_condition=ic_spec,
                golden_solution=golden_spec,
                diffusion_coefficient=diffusion_coefficient,
                decay_rate=decay_rate,
                boundary_condition=bc_spec,
                agents=agents_spec,
                bulk=bulk_spec,
                description=scenario_desc,
            )

        # ---- Build runner ----------------------------------------------------
        runner = BenchmarkRunner()
        for method_name in selected_methods:
            runner.add_schema(ALL_SCHEMAS[method_name], method_name)
        runner.add_scenario(scenario)

        # ---- Execute ---------------------------------------------------------
        log_area = st.empty()
        progress_bar = st.progress(0, text="Running benchmark...")

        start = time.time()
        buf = io.StringIO()
        with redirect_stdout(buf):
            results = runner.run(
                output_dir=output_dir,
                store_history=store_history,
                generate_plots=generate_plots,
            )
        elapsed = time.time() - start
        progress_bar.progress(100, text=f"Done in {elapsed:.2f}s")

        # Show log
        with st.expander("Console output"):
            st.text(buf.getvalue())

        # ---- Results ---------------------------------------------------------
        st.subheader("Results")

        for (schema_name, sc_name), res in results.items():
            with st.expander(f"{schema_name} on {sc_name}"):
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Duration", f"{res['duration']:.4f} s")
                if res.get("errors"):
                    col_m2.metric("L2 error", f"{res['errors'].get('l2_relative', 'N/A'):.6e}")
                    col_m3.metric("L∞ error", f"{res['errors'].get('linf_relative', 'N/A'):.6e}")
                else:
                    col_m2.metric("L2 error", "N/A")
                    col_m3.metric("L∞ error", "N/A")

                # Show figures
                if res.get("figures"):
                    for fig_path in res["figures"]:
                        if Path(fig_path).exists():
                            st.image(str(fig_path), use_container_width=True)

        # Summary table
        st.divider()
        st.subheader("Summary Table")
        summary_path = Path(output_dir) / "summary.csv"
        summary = runner.generate_summary_report(output_path=str(summary_path))
        if summary is not None:
            st.dataframe(summary, use_container_width=True)
            st.success(f"Summary saved to {summary_path}")
