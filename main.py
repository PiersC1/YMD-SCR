"""
Main Command-Line Interface (CLI) for the YMD Solver
Provides commands to solve vehicle configs, run 1D/2D parameter sweeps, and export reports.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config, VehicleConfig, LBF_PER_IN_TO_N_PER_M, MPH_TO_MS
from src.tire_model import load_nn_tire_model
from src.solver import YMDSolver
from src.kpi import KPICalculator
from src.visualization import YMDVisualizer


def print_kpi_table(kpis):
    print("\n" + "=" * 55)
    print("        YAW MOMENT DIAGRAM (YMD) PERFORMANCE KPIS")
    print("=" * 55)
    print(f" Velocity                     : {kpis.velocity_mph:.1f} mph")
    print(f" Steady-State Grip Limit (Mz=0): {kpis.steady_state_grip_limit_g:.3f} g ({kpis.steady_state_grip_limit_ms2:.2f} m/s²)")
    print(f" Max Lateral Acceleration     : {kpis.max_ay_g:.3f} g ({kpis.max_ay_ms2:.2f} m/s²)")
    print(f" Limit Balance (at max Ay)    : {kpis.limit_balance_nm:+.1f} N*m [{kpis.limit_balance_type}]")
    print(f" Control Authority (dMz/dδ)   : {kpis.control_nm_per_deg:+.2f} N*m/deg")
    print(f" Yaw Stability (dMz/dβ)       : {kpis.stability_nm_per_deg:+.2f} N*m/deg")
    print(f" Front TLLTD                  : {kpis.tlltd_front_percent:.1f} %")
    print(f" Roll Gradient                : {kpis.roll_gradient_deg_per_g:.2f} deg/g")
    print("=" * 55 + "\n")


def solve_single(args):
    config_path = args.config
    print(f"[*] Loading vehicle configuration: {config_path}")
    config = load_config(config_path)

    if args.velocity is not None:
        config.simulation.velocity = args.velocity * MPH_TO_MS
        config.simulation.velocity_display = args.velocity
        config.simulation.velocity_unit = "mph"

    print("[*] Initializing neural network tire model...")
    tire_model = load_nn_tire_model()

    print("[*] Running steady-state YMD solver...")
    solver = YMDSolver(config, tire_model=tire_model)
    result = solver.solve_grid()

    print("[*] Computing vehicle dynamics KPIs...")
    kpi_calc = KPICalculator(result)
    kpis = kpi_calc.compute_all_kpis(solver=solver)

    print_kpi_table(kpis)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export Matplotlib PNG
    png_path = out_dir / f"ymd_{kpis.velocity_mph:.0f}mph.png"
    YMDVisualizer.plot_matplotlib(result, kpis=kpis, save_path=png_path)
    print(f"[+] Saved high-resolution YMD plot: {png_path}")

    # Export Interactive Plotly HTML
    html_path = out_dir / f"ymd_{kpis.velocity_mph:.0f}mph.html"
    plotly_fig = YMDVisualizer.plot_plotly(result, kpis=kpis)
    plotly_fig.write_html(str(html_path))
    print(f"[+] Saved interactive Plotly YMD: {html_path}")

    # Export JSON KPI Report
    json_path = out_dir / f"kpi_report_{kpis.velocity_mph:.0f}mph.json"
    with open(json_path, "w") as f:
        json.dump(kpis.__dict__, f, indent=4)
    print(f"[+] Saved KPI summary report: {json_path}")


def run_sweep(args):
    config_path = args.config
    print(f"[*] Loading vehicle configuration: {config_path}")
    config = load_config(config_path)
    tire_model = load_nn_tire_model()
    solver = YMDSolver(config, tire_model=tire_model)

    param = args.parameter.lower()
    val_min = args.min
    val_max = args.max
    points = args.points
    values = np.linspace(val_min, val_max, points)

    from src.config import INCH_TO_M, LB_TO_KG, PSI_TO_PA

    CLI_SWEEPS = {
        "velocity": ("Velocity (mph)", lambda cfg, val: (setattr(cfg.simulation, "velocity", val * MPH_TO_MS), setattr(cfg.simulation, "velocity_display", val))),
        "speed": ("Velocity (mph)", lambda cfg, val: (setattr(cfg.simulation, "velocity", val * MPH_TO_MS), setattr(cfg.simulation, "velocity_display", val))),
        "front_spring": ("Front Spring Rate (lb/in)", lambda cfg, val: setattr(cfg.front_suspension, "spring_rate", val * LBF_PER_IN_TO_N_PER_M)),
        "rear_spring": ("Rear Spring Rate (lb/in)", lambda cfg, val: setattr(cfg.rear_suspension, "spring_rate", val * LBF_PER_IN_TO_N_PER_M)),
        "front_arb": ("Front ARB Stiffness (lb/in)", lambda cfg, val: setattr(cfg.front_suspension, "arb_stiffness", val * LBF_PER_IN_TO_N_PER_M)),
        "rear_arb": ("Rear ARB Stiffness (lb/in)", lambda cfg, val: setattr(cfg.rear_suspension, "arb_stiffness", val * LBF_PER_IN_TO_N_PER_M)),
        "cl": ("Aero Cl", lambda cfg, val: setattr(cfg.aero, "Cl", val)),
        "cop": ("Aero CoP (% Front)", lambda cfg, val: setattr(cfg.aero, "CoP", val)),
        "cg_front": ("Weight Distribution (% Front)", lambda cfg, val: setattr(cfg.mass, "x_loc_front", val / 100.0 if val > 1 else val)),
        "cg_height": ("CG Height (in)", lambda cfg, val: setattr(cfg.mass, "cg_height", val * INCH_TO_M)),
        "front_rc": ("Front Roll Center (in)", lambda cfg, val: setattr(cfg.front_suspension, "roll_center_height", val * INCH_TO_M)),
        "rear_rc": ("Rear Roll Center (in)", lambda cfg, val: setattr(cfg.rear_suspension, "roll_center_height", val * INCH_TO_M)),
        "front_camber": ("Front Static Camber (deg)", lambda cfg, val: setattr(cfg.front_suspension, "static_camber", np.deg2rad(val))),
        "rear_camber": ("Rear Static Camber (deg)", lambda cfg, val: setattr(cfg.rear_suspension, "static_camber", np.deg2rad(val))),
        "ackermann": ("Ackermann Steering (%)", lambda cfg, val: setattr(cfg.steering, "ackermann_percent", val)),
        "tire_pressure": ("Tire Pressure (psi)", lambda cfg, val: setattr(cfg.tire, "inflation_pressure", val * PSI_TO_PA)),
        "wheelbase": ("Wheelbase (in)", lambda cfg, val: setattr(cfg.dimensions, "wheelbase", val * INCH_TO_M)),
        "dry_mass": ("Dry Mass (lb)", lambda cfg, val: (setattr(cfg.mass, "dry_mass", val * LB_TO_KG), setattr(cfg.mass, "total_mass", val * LB_TO_KG + cfg.mass.driver_mass + cfg.mass.fuel_mass))),
    }

    if param not in CLI_SWEEPS:
        print(f"[-] Unknown sweep parameter: '{param}'.")
        print(f"[*] Available parameters: {', '.join(sorted(CLI_SWEEPS.keys()))}")
        sys.exit(1)

    display_name, setter = CLI_SWEEPS[param]
    sweep_res = solver.run_1d_sweep(setter, values, param_name=display_name)

    print("\n" + "=" * 70)
    print(f" 1D SWEEP SUMMARY: {sweep_res.param_names[0]}")
    print("=" * 70)
    print(f" {'Value':<12} | {'Grip Limit (g)':<15} | {'Limit Balance (N*m)':<20} | {'TLLTD (%)':<10}")
    print("-" * 70)
    for val, kpi in zip(values, sweep_res.kpis):
        print(f" {val:<12.2f} | {kpi.steady_state_grip_limit_g:<15.3f} | {kpi.limit_balance_nm:<20.1f} | {kpi.tlltd_front_percent:<10.1f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SCR Yaw Moment Diagram (YMD) Solver CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Solve command
    solve_p = subparsers.add_parser("solve", help="Solve YMD for a vehicle configuration")
    solve_p.add_argument("--config", "-c", type=str, default="base_params_SCR26.yaml", help="Path to YAML config")
    solve_p.add_argument("--velocity", "-v", type=float, default=None, help="Override speed in mph")
    solve_p.add_argument("--output-dir", "-o", type=str, default="outputs", help="Directory to save plots & KPIs")

    # Sweep command
    sweep_p = subparsers.add_parser("sweep", help="Run 1D parameter sweep")
    sweep_p.add_argument("--config", "-c", type=str, default="base_params_SCR26.yaml", help="Path to YAML config")
    sweep_p.add_argument("--parameter", "-p", type=str, required=True, help="Parameter (velocity, front_arb, cl, cop, cg_front)")
    sweep_p.add_argument("--min", type=float, required=True, help="Minimum parameter value")
    sweep_p.add_argument("--max", type=float, required=True, help="Maximum parameter value")
    sweep_p.add_argument("--points", "-n", type=int, default=5, help="Number of sweep points")
    sweep_p.add_argument("--output-dir", "-o", type=str, default="outputs", help="Directory to save sweep results")

    # GUI command
    subparsers.add_parser("gui", help="Launch interactive Streamlit Web GUI")

    args = parser.parse_args()

    if args.command == "gui":
        import subprocess
        app_path = ROOT_DIR / "src" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    elif args.command == "solve" or args.command is None:
        if args.command is None:
            # Default to solve if no command given
            args.config = "base_params_SCR26.yaml"
            args.velocity = None
            args.output_dir = "outputs"
        solve_single(args)
    elif args.command == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()
