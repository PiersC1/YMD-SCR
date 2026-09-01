"""
Streamlit Web GUI for the Yaw Moment Diagram (YMD) Solver
Provides interactive vehicle setup tuning, live YMD rendering, setup comparison, and parameter sweeps.
"""

import sys
from pathlib import Path

# Ensure repository root is in Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import copy
import io

from src.config import (
    load_config,
    VehicleConfig,
    MPH_TO_MS,
    MS_TO_MPH,
    LBF_PER_IN_TO_N_PER_M,
    N_PER_M_TO_LBF_PER_IN,
    LB_TO_KG,
    KG_TO_LB,
    INCH_TO_M,
    M_TO_INCH,
    PSI_TO_PA,
    LB_PER_FT3_TO_KG_PER_M3,
    DEG_TO_RAD,
    RAD_TO_DEG,
    G_ACCEL,
)
from src.tire_model import load_nn_tire_model
from src.solver import YMDSolver, YMDResult
from src.kpi import KPICalculator, VehicleKPIs
from src.visualization import YMDVisualizer


# Page setup
st.set_page_config(
    page_title="YMD Solver | SC Racing",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# USC Brand CSS Facelift (USC Cardinal: #990000, USC Gold: #FFCC00, Dark Gray: #5C5C5C, Light Gray: #C7C7C7)
st.markdown(
    """<style>
.usc-header-container {
    border-left: 6px solid #990000;
    padding: 2px 0 2px 14px;
    margin-bottom: 20px;
}
.usc-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #990000;
    margin: 0;
    letter-spacing: -0.5px;
    display: inline-block;
}
.usc-badge {
    background-color: #FFCC00;
    color: #990000;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.85rem;
    margin-left: 10px;
    vertical-align: middle;
    display: inline-block;
}
.usc-subtitle {
    font-size: 0.95rem;
    color: #5C5C5C;
    margin-top: 4px;
    font-weight: 500;
}
/* Custom Primary Button */
div.stButton > button[kind="primary"] {
    background-color: #990000 !important;
    color: #FFFFFF !important;
    border: 1.5px solid #FFCC00 !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #800000 !important;
    color: #FFCC00 !important;
    border: 1.5px solid #FFCC00 !important;
}
/* Metric cards with anti-truncation styling */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-top: 3.5px solid #990000;
    padding: 8px 10px !important;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    min-width: 0 !important;
}
[data-testid="stMetricLabel"] {
    color: #5C5C5C !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    white-space: normal !important;
    overflow: visible !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: #111111 !important;
    font-weight: 700 !important;
    font-size: 1.30rem !important;
    white-space: normal !important;
    overflow: visible !important;
}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: #5C5C5C !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #990000 !important;
    border-bottom-color: #990000 !important;
}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_tire_model():
    """Cache the PyTorch Neural Network tire model in memory."""
    weights = ROOT_DIR / "NN_model" / "R20_tire_model.pth"
    norm = ROOT_DIR / "NN_model" / "norm_params.json"
    return load_nn_tire_model(weights_path=str(weights), norm_params_path=str(norm))


def get_baseline_config():
    """Load default vehicle configuration fresh from disk without stale memory caching."""
    cfg_path = ROOT_DIR / "base_params_SCR26.yaml"
    return load_config(cfg_path)


def main():
    # USC Branded Clean Header (unindented to prevent markdown code block parsing)
    st.markdown(
        """<div class="usc-header-container">
    <div>
        <span class="usc-title">YMD Solver</span>
        <span class="usc-badge">SC RACING</span>
    </div>
    <div class="usc-subtitle">Steady-State Vehicle Dynamics & Handling Envelope</div>
</div>""",
        unsafe_allow_html=True,
    )

    tire_model = get_tire_model()
    base_config = get_baseline_config()

    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0
    rc = st.session_state["reset_counter"]

    # Sidebar: Vehicle Setup Controls
    st.sidebar.header("Vehicle Configuration")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("Reset Baseline", help="Reset all parameters back to default SCR26 values"):
        st.session_state["reset_counter"] += 1
        for k in ["active_result", "active_kpis", "active_config"]:
            st.session_state.pop(k, None)
        st.rerun()

    auto_solve = st.sidebar.checkbox("Auto-Solve on Change", value=False, help="Automatically re-run solver on every slider change. Leave unchecked to adjust multiple parameters and solve on demand.")

    solve_clicked = st.sidebar.button("Solve / Update YMD", type="primary", width="stretch", help="Compute steady-state YMD with current parameter settings")

    config_source = st.sidebar.radio(
        "Configuration Source",
        ["Default SCR26", "Upload Custom YAML"],
        index=0,
        key=f"cfg_source_{rc}",
    )

    if config_source == "Upload Custom YAML":
        uploaded_file = st.sidebar.file_uploader("Upload YAML Config", type=["yaml", "yml"], key=f"uploader_{rc}")
        if uploaded_file is not None:
            temp_path = ROOT_DIR / "temp_uploaded_config.yaml"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            try:
                active_config = load_config(temp_path)
            except Exception as e:
                st.sidebar.error(f"⚠️ Failed to parse uploaded YAML: {e}")
                active_config = copy.deepcopy(base_config)
        else:
            active_config = copy.deepcopy(base_config)
    else:
        active_config = copy.deepcopy(base_config)

    # 1. Operating & Solver Parameters
    with st.sidebar.expander("Operating & Grid Settings", expanded=True):
        speed_mph = st.slider(
            "Vehicle Speed (mph)",
            min_value=5.0,
            max_value=75.0,
            value=float(active_config.simulation.velocity_display),
            step=1.0,
            key=f"speed_{rc}",
        )
        active_config.simulation.velocity = speed_mph * MPH_TO_MS
        active_config.simulation.velocity_display = speed_mph

        st.markdown("**Grid Resolution**")
        b_min = st.number_input("Chassis Slip β Min (deg)", value=-15.0, step=1.0, key=f"b_min_{rc}")
        b_max = st.number_input("Chassis Slip β Max (deg)", value=15.0, step=1.0, key=f"b_max_{rc}")
        b_pts = st.number_input("Chassis Slip β Points", value=15, min_value=5, max_value=41, step=2, key=f"b_pts_{rc}")

        d_min = st.number_input("Steer Angle δ Min (deg)", value=-20.0, step=1.0, key=f"d_min_{rc}")
        d_max = st.number_input("Steer Angle δ Max (deg)", value=20.0, step=1.0, key=f"d_max_{rc}")
        d_pts = st.number_input("Steer Angle δ Points", value=21, min_value=5, max_value=51, step=2, key=f"d_pts_{rc}")

        active_config.simulation.beta_sweep.min_val = b_min
        active_config.simulation.beta_sweep.max_val = b_max
        active_config.simulation.beta_sweep.points = int(b_pts)

        active_config.simulation.delta_sweep.min_val = d_min
        active_config.simulation.delta_sweep.max_val = d_max
        active_config.simulation.delta_sweep.points = int(d_pts)

        damping = st.slider("Solver Damping Factor (λ)", min_value=0.1, max_value=0.9, value=0.35, step=0.05, key=f"damping_{rc}")
        active_config.simulation.damping_factor = damping

    # 2. Mass, CG & Dimensions
    with st.sidebar.expander("Mass, CG & Dimensions", expanded=False):
        c_m1, c_m2 = st.columns(2)
        dry_m_lb = c_m1.number_input("Dry Mass (lb)", value=float(active_config.mass.dry_mass * KG_TO_LB), step=5.0, key=f"dry_m_{rc}")
        driver_m_lb = c_m2.number_input("Driver Mass (lb)", value=float(active_config.mass.driver_mass * KG_TO_LB), step=5.0, key=f"driver_m_{rc}")
        fuel_m_lb = st.number_input("Fuel Mass (lb)", value=float(active_config.mass.fuel_mass * KG_TO_LB), step=1.0, key=f"fuel_m_{rc}")

        total_m_kg = (dry_m_lb + driver_m_lb + fuel_m_lb) * LB_TO_KG
        active_config.mass.dry_mass = dry_m_lb * LB_TO_KG
        active_config.mass.driver_mass = driver_m_lb * LB_TO_KG
        active_config.mass.fuel_mass = fuel_m_lb * LB_TO_KG
        active_config.mass.total_mass = total_m_kg

        st.caption(f"Total Mass: **{total_m_kg * KG_TO_LB:.1f} lb** ({total_m_kg:.1f} kg)")

        cg_pct_front = st.slider("Weight Dist (% Front)", min_value=30.0, max_value=70.0, value=float(active_config.mass.x_loc_front * 100.0), step=0.5, key=f"cg_front_{rc}")
        active_config.mass.x_loc_front = cg_pct_front / 100.0

        cg_pct_left = st.slider("Weight Dist (% Left)", min_value=40.0, max_value=60.0, value=float(active_config.mass.y_loc_left * 100.0), step=0.1, key=f"cg_left_{rc}")
        active_config.mass.y_loc_left = cg_pct_left / 100.0

        cg_h_in = st.number_input("CG Height (in)", value=float(active_config.mass.cg_height * M_TO_INCH), step=0.25, key=f"cg_h_{rc}")
        active_config.mass.cg_height = cg_h_in * INCH_TO_M

        wheelbase_in = st.number_input("Wheelbase (in)", value=float(active_config.dimensions.wheelbase * M_TO_INCH), step=0.5, key=f"wb_{rc}")
        active_config.dimensions.wheelbase = wheelbase_in * INCH_TO_M

    # 3. Aerodynamics
    with st.sidebar.expander("Aerodynamics", expanded=False):
        cl_val = st.number_input("Lift/Downforce Coeff (Cl)", value=float(active_config.aero.Cl), step=0.1, key=f"cl_{rc}")
        cop_val = st.slider("Aero Balance (CoP % Front)", min_value=15.0, max_value=85.0, value=float(active_config.aero.CoP), step=0.5, key=f"cop_{rc}")
        area_val = st.number_input("Frontal Area (m²)", value=float(active_config.aero.frontal_area), step=0.05, key=f"area_{rc}")
        rho_val = st.number_input("Air Density (kg/m³)", value=float(active_config.aero.rho), step=0.01, key=f"rho_{rc}")

        active_config.aero.Cl = cl_val
        active_config.aero.CoP = cop_val
        active_config.aero.frontal_area = area_val
        active_config.aero.rho = rho_val

    # 4. Front Suspension
    with st.sidebar.expander("Front Suspension & Geometry", expanded=False):
        f_k_s = st.number_input("Front Spring Rate (lb/in)", value=float(active_config.front_suspension.spring_rate / LBF_PER_IN_TO_N_PER_M), step=25.0, key=f"f_k_s_{rc}")
        f_k_arb = st.number_input("Front ARB Stiffness (lb/in)", value=float(active_config.front_suspension.arb_stiffness / LBF_PER_IN_TO_N_PER_M), step=25.0, key=f"f_k_arb_{rc}")
        f_tw = st.number_input("Front Track Width (in)", value=float(active_config.front_suspension.track_width * M_TO_INCH), step=0.5, key=f"f_tw_{rc}")
        f_rc = st.number_input("Front Roll Center Height (in)", value=float(active_config.front_suspension.roll_center_height * M_TO_INCH), step=0.25, key=f"f_rc_{rc}")
        f_smr = st.number_input("Front Spring Motion Ratio", value=float(active_config.front_suspension.spring_MR), step=0.05, key=f"f_smr_{rc}")
        f_amr = st.number_input("Front ARB Motion Ratio", value=float(active_config.front_suspension.arb_MR), step=0.05, key=f"f_amr_{rc}")
        f_camber = st.number_input("Front Static Camber (deg)", value=float(np.rad2deg(active_config.front_suspension.static_camber)), step=0.25, key=f"f_camber_{rc}")
        f_toe = st.number_input("Front Static Toe (deg, - = out)", value=float(np.rad2deg(active_config.front_suspension.static_toe)), step=0.1, key=f"f_toe_{rc}")
        f_umass = st.number_input("Front Corner Unsprung Mass (lb)", value=float(active_config.front_suspension.unsprung_mass * KG_TO_LB), step=1.0, key=f"f_umass_{rc}")
        f_ucg = st.number_input("Front Unsprung CG Height (in)", value=float(active_config.front_suspension.unsprung_cg_height * M_TO_INCH), step=0.5, key=f"f_ucg_{rc}")

        active_config.front_suspension.spring_rate = f_k_s * LBF_PER_IN_TO_N_PER_M
        active_config.front_suspension.arb_stiffness = f_k_arb * LBF_PER_IN_TO_N_PER_M
        active_config.front_suspension.track_width = f_tw * INCH_TO_M
        active_config.front_suspension.roll_center_height = f_rc * INCH_TO_M
        active_config.front_suspension.spring_MR = f_smr
        active_config.front_suspension.arb_MR = f_amr
        active_config.front_suspension.static_camber = np.deg2rad(f_camber)
        active_config.front_suspension.static_toe = np.deg2rad(f_toe)
        active_config.front_suspension.unsprung_mass = f_umass * LB_TO_KG
        active_config.front_suspension.unsprung_cg_height = f_ucg * INCH_TO_M

    # 5. Rear Suspension
    with st.sidebar.expander("Rear Suspension & Geometry", expanded=False):
        r_k_s = st.number_input("Rear Spring Rate (lb/in)", value=float(active_config.rear_suspension.spring_rate / LBF_PER_IN_TO_N_PER_M), step=25.0, key=f"r_k_s_{rc}")
        r_k_arb = st.number_input("Rear ARB Stiffness (lb/in)", value=float(active_config.rear_suspension.arb_stiffness / LBF_PER_IN_TO_N_PER_M), step=25.0, key=f"r_k_arb_{rc}")
        r_tw = st.number_input("Rear Track Width (in)", value=float(active_config.rear_suspension.track_width * M_TO_INCH), step=0.5, key=f"r_tw_{rc}")
        r_rc = st.number_input("Rear Roll Center Height (in)", value=float(active_config.rear_suspension.roll_center_height * M_TO_INCH), step=0.25, key=f"r_rc_{rc}")
        r_smr = st.number_input("Rear Spring Motion Ratio", value=float(active_config.rear_suspension.spring_MR), step=0.05, key=f"r_smr_{rc}")
        r_amr = st.number_input("Rear ARB Motion Ratio", value=float(active_config.rear_suspension.arb_MR), step=0.05, key=f"r_amr_{rc}")
        r_camber = st.number_input("Rear Static Camber (deg)", value=float(np.rad2deg(active_config.rear_suspension.static_camber)), step=0.25, key=f"r_camber_{rc}")
        r_toe = st.number_input("Rear Static Toe (deg, + = in)", value=float(np.rad2deg(active_config.rear_suspension.static_toe)), step=0.1, key=f"r_toe_{rc}")
        r_umass = st.number_input("Rear Corner Unsprung Mass (lb)", value=float(active_config.rear_suspension.unsprung_mass * KG_TO_LB), step=1.0, key=f"r_umass_{rc}")
        r_ucg = st.number_input("Rear Unsprung CG Height (in)", value=float(active_config.rear_suspension.unsprung_cg_height * M_TO_INCH), step=0.5, key=f"r_ucg_{rc}")

        active_config.rear_suspension.spring_rate = r_k_s * LBF_PER_IN_TO_N_PER_M
        active_config.rear_suspension.arb_stiffness = r_k_arb * LBF_PER_IN_TO_N_PER_M
        active_config.rear_suspension.track_width = r_tw * INCH_TO_M
        active_config.rear_suspension.roll_center_height = r_rc * INCH_TO_M
        active_config.rear_suspension.spring_MR = r_smr
        active_config.rear_suspension.arb_MR = r_amr
        active_config.rear_suspension.static_camber = np.deg2rad(r_camber)
        active_config.rear_suspension.static_toe = np.deg2rad(r_toe)
        active_config.rear_suspension.unsprung_mass = r_umass * LB_TO_KG
        active_config.rear_suspension.unsprung_cg_height = r_ucg * INCH_TO_M

    # 6. Steering & Tires
    with st.sidebar.expander("Steering & Tires", expanded=False):
        ack_pct = st.slider("Ackermann Steering (%)", min_value=0.0, max_value=150.0, value=float(active_config.steering.ackermann_percent), step=5.0, key=f"ack_{rc}")
        tire_p_psi = st.number_input("Tire Inflation Pressure (psi)", value=float(active_config.tire.inflation_pressure / PSI_TO_PA), step=0.5, key=f"tire_p_{rc}")
        tire_k_lbin = st.number_input("Tire Radial Stiffness (lb/in)", value=float(active_config.tire.tire_stiffness / LBF_PER_IN_TO_N_PER_M), step=25.0, key=f"tire_k_{rc}")
        tire_scale = st.slider("Tire Grip Multiplier / Scale", min_value=0.70, max_value=1.30, value=0.93, step=0.01, key=f"tire_scale_{rc}")

        active_config.steering.ackermann_percent = ack_pct
        active_config.tire.inflation_pressure = tire_p_psi * PSI_TO_PA
        active_config.tire.tire_stiffness = tire_k_lbin * LBF_PER_IN_TO_N_PER_M
        tire_model.scale_factor = tire_scale

    # Tabs for main interface
    tab_ymd, tab_sweep, tab_export = st.tabs(["YMD & Handling KPIs", "Sensitivity Sweeps", "Export & Config"])

    # Solve logic with session state caching
    should_solve = (
        "active_result" not in st.session_state
        or solve_clicked
        or auto_solve
    )

    if should_solve:
        with st.spinner("Solving Yaw Moment Diagram equilibrium..."):
            solver = YMDSolver(active_config, tire_model=tire_model)
            result = solver.solve_grid()
            kpi_calc = KPICalculator(result)
            kpis = kpi_calc.compute_all_kpis(solver=solver)
            st.session_state["active_result"] = result
            st.session_state["active_kpis"] = kpis
            st.session_state["active_config"] = active_config
    else:
        result = st.session_state["active_result"]
        kpis = st.session_state["active_kpis"]
        solver = YMDSolver(active_config, tire_model=tire_model)

    if not auto_solve and not solve_clicked and "active_result" in st.session_state:
        st.info("💡 Adjust your desired sliders in the sidebar, then click **'🚀 Solve / Update YMD'** to compute. (Or turn on **'⚡ Auto-Solve on Change'**).")

    with tab_ymd:
        # Top KPI Metrics Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Steady Grip Limit", f"{kpis.steady_state_grip_limit_g:.2f} g", help="Maximum Ay where Mz = 0")
        c2.metric("Max Lateral Accel", f"{kpis.max_ay_g:.2f} g", help="Absolute peak Ay reached")
        c3.metric("Limit Balance", f"{kpis.limit_balance_nm:+.0f} N*m", kpis.limit_balance_type, help="Yaw moment at max Ay (<0 = understeer, >0 = oversteer)")
        c4.metric("Control Authority", f"{kpis.control_nm_per_deg:+.1f}", "N*m/deg", help="dMz/dDelta at (0,0)")
        c5.metric("Yaw Stability", f"{kpis.stability_nm_per_deg:+.1f}", "N*m/deg", help="dMz/dBeta at (0,0)")
        c6.metric("Front TLLTD", f"{kpis.tlltd_front_percent:.1f} %", help="Front lateral load transfer fraction")

        # Baseline Comparison Option
        enable_comp = st.checkbox("Overlay Default Baseline Setup", value=False)
        comp_res = None
        if enable_comp:
            comp_cfg = copy.deepcopy(base_config)
            comp_cfg.simulation.velocity = active_config.simulation.velocity
            comp_cfg.simulation.velocity_display = active_config.simulation.velocity_display
            comp_cfg.simulation.beta_sweep = copy.deepcopy(active_config.simulation.beta_sweep)
            comp_cfg.simulation.delta_sweep = copy.deepcopy(active_config.simulation.delta_sweep)
            base_solver = YMDSolver(comp_cfg, tire_model=tire_model)
            comp_res = base_solver.solve_grid()

        # Plotly YMD Figure
        plotly_fig = YMDVisualizer.plot_plotly(
            result,
            kpis=kpis,
            title=f"SCR Yaw Moment Diagram @ {speed_mph:.0f} mph",
            compare_result=comp_res,
            compare_label="Default Baseline",
        )
        st.plotly_chart(plotly_fig, width="stretch")

        # Detailed state inspector
        with st.expander("Inspect Operating Points & 4-Corner Tire States"):
            beta_choices = result.beta_grid_deg[:, 0]
            delta_choices = result.delta_grid_deg[0, :]

            sel_c1, sel_c2 = st.columns(2)
            sel_beta = sel_c1.selectbox("Select Body Slip Angle β (deg)", beta_choices, index=len(beta_choices)//2)
            sel_delta = sel_c2.selectbox("Select Steer Angle δ (deg)", delta_choices, index=len(delta_choices)//2)

            b_idx = int(np.argmin(np.abs(beta_choices - sel_beta)))
            d_idx = int(np.argmin(np.abs(delta_choices - sel_delta)))
            sol_pt = result.point_solutions[b_idx][d_idx]

            st.write(f"**Operating State**: Ay = `{sol_pt.ay_g:.3f} g` (`{sol_pt.ay:.2f} m/s²`), Yaw Moment Mz = `{sol_pt.mz:.1f} N*m`, Yaw Rate = `{sol_pt.yaw_rate:.2f} rad/s`")

            corner_df = pd.DataFrame({
                "Corner": ["Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)"],
                "Normal Load Fz [N]": [sol_pt.corner_loads.FL, sol_pt.corner_loads.FR, sol_pt.corner_loads.RL, sol_pt.corner_loads.RR],
                "Slip Angle α [deg]": [
                    np.rad2deg(sol_pt.kinematics.FL.slip_angle),
                    np.rad2deg(sol_pt.kinematics.FR.slip_angle),
                    np.rad2deg(sol_pt.kinematics.RL.slip_angle),
                    np.rad2deg(sol_pt.kinematics.RR.slip_angle),
                ],
                "Lateral Force Fy [N]": [sol_pt.fy_corners["FL"], sol_pt.fy_corners["FR"], sol_pt.fy_corners["RL"], sol_pt.fy_corners["RR"]],
                "Aligning Torque Mz [N*m]": [sol_pt.mz_corners["FL"], sol_pt.mz_corners["FR"], sol_pt.mz_corners["RL"], sol_pt.mz_corners["RR"]],
            })
            st.dataframe(corner_df, width="stretch")

    SWEEP_SPECS = {
        # Operating & Solver
        "Vehicle Speed (mph)": {
            "category": "Operating", "min": 15.0, "max": 60.0, "pts": 10, "key": "sw_speed",
            "setter": lambda cfg, val: (setattr(cfg.simulation, "velocity", val * MPH_TO_MS), setattr(cfg.simulation, "velocity_display", val))
        },
        "Solver Damping Factor (λ)": {
            "category": "Operating", "min": 0.15, "max": 0.75, "pts": 7, "key": "sw_damping",
            "setter": lambda cfg, val: setattr(cfg.simulation, "damping_factor", val)
        },
        # Mass & CG
        "Weight Distribution (% Front)": {
            "category": "Mass & CG", "min": 40.0, "max": 60.0, "pts": 9, "key": "sw_wdf",
            "setter": lambda cfg, val: setattr(cfg.mass, "x_loc_front", val / 100.0)
        },
        "Weight Distribution (% Left)": {
            "category": "Mass & CG", "min": 45.0, "max": 55.0, "pts": 7, "key": "sw_wdl",
            "setter": lambda cfg, val: setattr(cfg.mass, "y_loc_left", val / 100.0)
        },
        "CG Height (in)": {
            "category": "Mass & CG", "min": 9.0, "max": 16.0, "pts": 8, "key": "sw_cgh",
            "setter": lambda cfg, val: setattr(cfg.mass, "cg_height", val * INCH_TO_M)
        },
        "Dry Mass (lb)": {
            "category": "Mass & CG", "min": 400.0, "max": 600.0, "pts": 9, "key": "sw_dry_m",
            "setter": lambda cfg, val: (setattr(cfg.mass, "dry_mass", val * LB_TO_KG), setattr(cfg.mass, "total_mass", val * LB_TO_KG + cfg.mass.driver_mass + cfg.mass.fuel_mass))
        },
        "Driver Mass (lb)": {
            "category": "Mass & CG", "min": 120.0, "max": 220.0, "pts": 6, "key": "sw_driver_m",
            "setter": lambda cfg, val: (setattr(cfg.mass, "driver_mass", val * LB_TO_KG), setattr(cfg.mass, "total_mass", cfg.mass.dry_mass + val * LB_TO_KG + cfg.mass.fuel_mass))
        },
        "Wheelbase (in)": {
            "category": "Mass & CG", "min": 55.0, "max": 66.0, "pts": 8, "key": "sw_wb",
            "setter": lambda cfg, val: setattr(cfg.dimensions, "wheelbase", val * INCH_TO_M)
        },
        # Aerodynamics
        "Aero Downforce (Cl)": {
            "category": "Aerodynamics", "min": 1.0, "max": 5.5, "pts": 9, "key": "sw_cl",
            "setter": lambda cfg, val: setattr(cfg.aero, "Cl", val)
        },
        "Aero Balance (CoP % Front)": {
            "category": "Aerodynamics", "min": 25.0, "max": 65.0, "pts": 9, "key": "sw_cop",
            "setter": lambda cfg, val: setattr(cfg.aero, "CoP", val)
        },
        "Frontal Area (m²)": {
            "category": "Aerodynamics", "min": 0.8, "max": 1.6, "pts": 9, "key": "sw_area",
            "setter": lambda cfg, val: setattr(cfg.aero, "frontal_area", val)
        },
        "Air Density (kg/m³)": {
            "category": "Aerodynamics", "min": 1.05, "max": 1.30, "pts": 6, "key": "sw_rho",
            "setter": lambda cfg, val: setattr(cfg.aero, "rho", val)
        },
        # Front Suspension
        "Front Spring Rate (lb/in)": {
            "category": "Front Suspension", "min": 150.0, "max": 600.0, "pts": 9, "key": "sw_fspring",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "spring_rate", val * LBF_PER_IN_TO_N_PER_M)
        },
        "Front ARB Stiffness (lb/in)": {
            "category": "Front Suspension", "min": 0.0, "max": 400.0, "pts": 9, "key": "sw_farb",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "arb_stiffness", val * LBF_PER_IN_TO_N_PER_M)
        },
        "Front Roll Center Height (in)": {
            "category": "Front Suspension", "min": 0.5, "max": 6.0, "pts": 8, "key": "sw_frc",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "roll_center_height", val * INCH_TO_M)
        },
        "Front Track Width (in)": {
            "category": "Front Suspension", "min": 42.0, "max": 54.0, "pts": 7, "key": "sw_ftw",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "track_width", val * INCH_TO_M)
        },
        "Front Static Camber (deg)": {
            "category": "Front Suspension", "min": -4.0, "max": 1.0, "pts": 9, "key": "sw_fcamber",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "static_camber", np.deg2rad(val))
        },
        "Front Static Toe (deg)": {
            "category": "Front Suspension", "min": -2.0, "max": 2.0, "pts": 9, "key": "sw_ftoe",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "static_toe", np.deg2rad(val))
        },
        "Front Spring Motion Ratio": {
            "category": "Front Suspension", "min": 0.70, "max": 1.40, "pts": 8, "key": "sw_fsmr",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "spring_MR", val)
        },
        "Front ARB Motion Ratio": {
            "category": "Front Suspension", "min": 0.0, "max": 1.30, "pts": 7, "key": "sw_famr",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "arb_MR", val)
        },
        "Front Unsprung Mass (lb)": {
            "category": "Front Suspension", "min": 6.0, "max": 18.0, "pts": 7, "key": "sw_fumass",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "unsprung_mass", val * LB_TO_KG)
        },
        "Front Unsprung CG Height (in)": {
            "category": "Front Suspension", "min": 5.0, "max": 12.0, "pts": 8, "key": "sw_fucg",
            "setter": lambda cfg, val: setattr(cfg.front_suspension, "unsprung_cg_height", val * INCH_TO_M)
        },
        # Rear Suspension
        "Rear Spring Rate (lb/in)": {
            "category": "Rear Suspension", "min": 150.0, "max": 600.0, "pts": 9, "key": "sw_rspring",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "spring_rate", val * LBF_PER_IN_TO_N_PER_M)
        },
        "Rear ARB Stiffness (lb/in)": {
            "category": "Rear Suspension", "min": 0.0, "max": 400.0, "pts": 9, "key": "sw_rarb",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "arb_stiffness", val * LBF_PER_IN_TO_N_PER_M)
        },
        "Rear Roll Center Height (in)": {
            "category": "Rear Suspension", "min": 0.5, "max": 6.0, "pts": 8, "key": "sw_rrc",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "roll_center_height", val * INCH_TO_M)
        },
        "Rear Track Width (in)": {
            "category": "Rear Suspension", "min": 42.0, "max": 54.0, "pts": 7, "key": "sw_rtw",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "track_width", val * INCH_TO_M)
        },
        "Rear Static Camber (deg)": {
            "category": "Rear Suspension", "min": -4.0, "max": 1.0, "pts": 9, "key": "sw_rcamber",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "static_camber", np.deg2rad(val))
        },
        "Rear Static Toe (deg)": {
            "category": "Rear Suspension", "min": -2.0, "max": 2.0, "pts": 9, "key": "sw_rtoe",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "static_toe", np.deg2rad(val))
        },
        "Rear Spring Motion Ratio": {
            "category": "Rear Suspension", "min": 0.70, "max": 1.40, "pts": 8, "key": "sw_rsmr",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "spring_MR", val)
        },
        "Rear ARB Motion Ratio": {
            "category": "Rear Suspension", "min": 0.0, "max": 1.30, "pts": 7, "key": "sw_ramr",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "arb_MR", val)
        },
        "Rear Unsprung Mass (lb)": {
            "category": "Rear Suspension", "min": 6.0, "max": 20.0, "pts": 8, "key": "sw_rumass",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "unsprung_mass", val * LB_TO_KG)
        },
        "Rear Unsprung CG Height (in)": {
            "category": "Rear Suspension", "min": 5.0, "max": 12.0, "pts": 8, "key": "sw_rucg",
            "setter": lambda cfg, val: setattr(cfg.rear_suspension, "unsprung_cg_height", val * INCH_TO_M)
        },
        # Steering & Tires
        "Ackermann Steering (%)": {
            "category": "Steering & Tires", "min": 0.0, "max": 140.0, "pts": 8, "key": "sw_ack",
            "setter": lambda cfg, val: setattr(cfg.steering, "ackermann_percent", val)
        },
        "Tire Inflation Pressure (psi)": {
            "category": "Steering & Tires", "min": 8.0, "max": 16.0, "pts": 9, "key": "sw_tire_p",
            "setter": lambda cfg, val: setattr(cfg.tire, "inflation_pressure", val * PSI_TO_PA)
        },
        "Tire Radial Stiffness (lb/in)": {
            "category": "Steering & Tires", "min": 300.0, "max": 650.0, "pts": 8, "key": "sw_tire_k",
            "setter": lambda cfg, val: setattr(cfg.tire, "tire_stiffness", val * LBF_PER_IN_TO_N_PER_M)
        },
    }

    with tab_sweep:
        st.subheader("1D Parameter Sensitivity Analysis")

        col_cat, col_param = st.columns([1, 2])
        categories = ["All", "Operating", "Mass & CG", "Aerodynamics", "Front Suspension", "Rear Suspension", "Steering & Tires"]
        selected_cat = col_cat.selectbox("Filter Category", categories, index=0)

        if selected_cat == "All":
            param_options = list(SWEEP_SPECS.keys())
        else:
            param_options = [k for k, v in SWEEP_SPECS.items() if v["category"] == selected_cat]

        sweep_var = col_param.selectbox("Select Parameter to Sweep", param_options)

        spec = SWEEP_SPECS[sweep_var]
        sweep_cols = st.columns(3)
        sw_min = sweep_cols[0].number_input(f"Min ({sweep_var.split('(')[-1].replace(')', '') if '(' in sweep_var else ''})", value=float(spec["min"]), key=f"{spec['key']}_min")
        sw_max = sweep_cols[1].number_input(f"Max ({sweep_var.split('(')[-1].replace(')', '') if '(' in sweep_var else ''})", value=float(spec["max"]), key=f"{spec['key']}_max")
        sw_pts = sweep_cols[2].number_input("Points", value=int(spec["pts"]), min_value=3, max_value=35, key=f"{spec['key']}_pts")

        sw_vals = np.linspace(sw_min, sw_max, int(sw_pts))
        sweep_func = spec["setter"]

        run_sweep_btn = st.button("Run Sensitivity Sweep", type="primary")

        if run_sweep_btn:
            with st.spinner("Solving parameter sweep across full YMD envelopes..."):
                sw_solver = YMDSolver(active_config, tire_model=tire_model)
                sw_res = sw_solver.run_1d_sweep(sweep_func, sw_vals, param_name=sweep_var)

                sw_df = pd.DataFrame({
                    sweep_var: sw_vals,
                    "Grip Limit [g]": [k.steady_state_grip_limit_g for k in sw_res.kpis],
                    "Max Lateral Accel [g]": [k.max_ay_g for k in sw_res.kpis],
                    "Limit Balance [N*m]": [k.limit_balance_nm for k in sw_res.kpis],
                    "Control Authority [N*m/deg]": [k.control_nm_per_deg for k in sw_res.kpis],
                    "Yaw Stability [N*m/deg]": [k.stability_nm_per_deg for k in sw_res.kpis],
                    "Front TLLTD [%]": [k.tlltd_front_percent for k in sw_res.kpis],
                })
                st.session_state["last_sweep_df"] = sw_df
                st.session_state["last_sweep_var"] = sweep_var

        if "last_sweep_df" in st.session_state and "last_sweep_var" in st.session_state:
            saved_df = st.session_state["last_sweep_df"]
            saved_var = st.session_state["last_sweep_var"]

            st.dataframe(saved_df, width="stretch")

            st.markdown("#### Performance Trends")

            tr_c1, tr_c2 = st.columns(2)

            with tr_c1:
                st.markdown("**Grip & Peak Lateral Acceleration [g]**")
                st.line_chart(saved_df.set_index(saved_var)[["Grip Limit [g]", "Max Lateral Accel [g]"]])

                st.markdown("**Steering Control & Yaw Stability [N*m/deg]**")
                st.line_chart(saved_df.set_index(saved_var)[["Control Authority [N*m/deg]", "Yaw Stability [N*m/deg]"]])

            with tr_c2:
                st.markdown("**Limit Balance [N*m] (Understeer < 0 < Oversteer)**")
                import plotly.graph_objects as go
                fig_lb = go.Figure()
                fig_lb.add_trace(go.Scatter(
                    x=saved_df[saved_var],
                    y=saved_df["Limit Balance [N*m]"],
                    mode="lines+markers",
                    name="Limit Balance",
                    line=dict(color="#990000", width=3),
                    marker=dict(size=8, color="#FFCC00", line=dict(color="#990000", width=1.5)),
                ))
                fig_lb.add_hline(y=0, line_dash="dash", line_color="#5C5C5C", annotation_text="Neutral Balance (0 N*m)", annotation_position="bottom right")
                fig_lb.update_layout(
                    xaxis_title=saved_var,
                    yaxis_title="Limit Balance Mz [N*m]",
                    height=280,
                    margin=dict(l=40, r=20, t=20, b=40),
                    template="plotly_white",
                )
                st.plotly_chart(fig_lb, width="stretch")

                st.markdown("**Front Lateral Load Transfer (TLLTD) [%]**")
                st.line_chart(saved_df.set_index(saved_var)[["Front TLLTD [%]"]])

    with tab_export:
        st.subheader("Export Visualizations, Data & Custom YAML Config")

        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.markdown("#### Export Plots & Grid Data")
            # Matplotlib PNG on-demand generation
            if st.button("Generate Publication Plot (300 DPI PNG)"):
                with st.spinner("Rendering high-res 300 DPI diagram..."):
                    try:
                        fig_mpl = YMDVisualizer.plot_matplotlib(result, kpis=kpis)
                        img_buffer = io.BytesIO()
                        fig_mpl.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
                        img_buffer.seek(0)
                        plt.close(fig_mpl)
                        st.session_state["saved_png_bytes"] = img_buffer.getvalue()
                    except Exception as e:
                        st.warning(f"Matplotlib export unavailable: {e}")

            if "saved_png_bytes" in st.session_state:
                st.download_button(
                    label="Download Generated PNG",
                    data=st.session_state["saved_png_bytes"],
                    file_name=f"YMD_plot_{speed_mph:.0f}mph.png",
                    mime="image/png",
                )

            # CSV Data Export
            flat_data = []
            for i in range(result.ay_grid_g.shape[0]):
                for j in range(result.ay_grid_g.shape[1]):
                    pt = result.point_solutions[i][j]
                    flat_data.append({
                        "beta_deg": result.beta_grid_deg[i, j],
                        "delta_deg": result.delta_grid_deg[i, j],
                        "ay_g": result.ay_grid_g[i, j],
                        "ay_ms2": result.ay_grid_ms2[i, j],
                        "mz_nm": result.mz_grid[i, j],
                        "yaw_rate_rad_s": result.yaw_rate_grid[i, j],
                        "FL_fz_N": pt.corner_loads.FL,
                        "FR_fz_N": pt.corner_loads.FR,
                        "RL_fz_N": pt.corner_loads.RL,
                        "RR_fz_N": pt.corner_loads.RR,
                    })
            csv_df = pd.DataFrame(flat_data)
            csv_buffer = csv_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Solved Grid CSV",
                data=csv_buffer,
                file_name=f"ymd_grid_data_{speed_mph:.0f}mph.csv",
                mime="text/csv",
            )

        with col_ex2:
            st.markdown("#### 📄 Export Tuned YAML Setup")
            # Generate YAML dictionary representing current active config
            tuned_yaml_data = {
                "aero_params": {
                    "Cl": float(active_config.aero.Cl),
                    "CoP": float(active_config.aero.CoP),
                    "rho": {"value": float(active_config.aero.rho * (1.0 / LB_PER_FT3_TO_KG_PER_M3)), "unit": "lb/ft^3"},
                    "frontal_area": {"value": float(active_config.aero.frontal_area), "unit": "m^2"},
                },
                "mass": {
                    "dry_mass": {"value": float(active_config.mass.dry_mass * KG_TO_LB), "unit": "lb"},
                    "driver_mass": {"value": float(active_config.mass.driver_mass * KG_TO_LB), "unit": "lb"},
                    "fuel_mass": {"value": float(active_config.mass.fuel_mass * KG_TO_LB), "unit": "lb"},
                    "x_loc": float(active_config.mass.x_loc_front),
                    "y_loc": float(active_config.mass.y_loc_left),
                    "z_loc": {"value": float(active_config.mass.cg_height * M_TO_INCH), "unit": "in"},
                },
                "dimensions": {
                    "wheelbase": {"value": float(active_config.dimensions.wheelbase * M_TO_INCH), "unit": "in"},
                },
                "frontSuspension": {
                    "stiffness": {
                        "spring_rate": {"value": float(active_config.front_suspension.spring_rate / LBF_PER_IN_TO_N_PER_M), "unit": "lb/in"},
                        "arb_stiffness": {"value": float(active_config.front_suspension.arb_stiffness / LBF_PER_IN_TO_N_PER_M), "unit": "lb/in"},
                    },
                    "geom": {
                        "track_width": {"value": float(active_config.front_suspension.track_width * M_TO_INCH), "unit": "in"},
                        "static_IA": {"value": float(np.rad2deg(active_config.front_suspension.static_camber)), "unit": "deg"},
                        "static_roll_center": {"value": float(active_config.front_suspension.roll_center_height * M_TO_INCH), "units": "in"},
                        "static_toe": {"value": float(np.rad2deg(active_config.front_suspension.static_toe)), "units": "deg"},
                    },
                    "kinematics": {
                        "spring_MR": float(active_config.front_suspension.spring_MR),
                        "arb_MR": float(active_config.front_suspension.arb_MR),
                    },
                    "mass": {
                        "unsprung_mass": {"value": float(active_config.front_suspension.unsprung_mass * KG_TO_LB), "unit": "lbs"},
                        "cg_height": {"value": float(active_config.front_suspension.unsprung_cg_height * M_TO_INCH), "unit": "in"},
                    },
                },
                "rearSuspension": {
                    "stiffness": {
                        "spring_rate": {"value": float(active_config.rear_suspension.spring_rate / LBF_PER_IN_TO_N_PER_M), "unit": "lb/in"},
                        "arb_stiffness": {"value": float(active_config.rear_suspension.arb_stiffness / LBF_PER_IN_TO_N_PER_M), "unit": "lb/in"},
                    },
                    "geom": {
                        "track_width": {"value": float(active_config.rear_suspension.track_width * M_TO_INCH), "unit": "in"},
                        "static_IA": {"value": float(np.rad2deg(active_config.rear_suspension.static_camber)), "unit": "deg"},
                        "static_roll_center": {"value": float(active_config.rear_suspension.roll_center_height * M_TO_INCH), "units": "in"},
                        "static_toe": {"value": float(np.rad2deg(active_config.rear_suspension.static_toe)), "units": "deg"},
                    },
                    "kinematics": {
                        "spring_MR": float(active_config.rear_suspension.spring_MR),
                        "arb_MR": float(active_config.rear_suspension.arb_MR),
                    },
                    "mass": {
                        "unsprung_mass": {"value": float(active_config.rear_suspension.unsprung_mass * KG_TO_LB), "unit": "lbs"},
                        "cg_height": {"value": float(active_config.rear_suspension.unsprung_cg_height * M_TO_INCH), "unit": "in"},
                    },
                },
                "steering": {
                    "ackermann": {"value": float(active_config.steering.ackermann_percent), "unit": "%"},
                },
                "tire_params": {
                    "inflation_pres": {"value": float(active_config.tire.inflation_pressure / PSI_TO_PA), "unit": "psi"},
                    "tire_stiffness": {"value": float(active_config.tire.tire_stiffness / LBF_PER_IN_TO_N_PER_M), "unit": "lb/in"},
                },
            }
            import yaml
            yaml_str = yaml.dump(tuned_yaml_data, default_flow_style=False, sort_keys=False)

            st.download_button(
                label="📥 Download Config as .YAML",
                data=yaml_str,
                file_name="tuned_vehicle_setup.yaml",
                mime="text/yaml",
            )


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as streamlit_runtime_exists
        if streamlit_runtime_exists():
            main()
        else:
            import subprocess
            print("[*] Starting Streamlit GUI server...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    except ImportError:
        main()
