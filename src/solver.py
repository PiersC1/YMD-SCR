"""
Modular Steady-State Yaw Moment Diagram (YMD) Solver Engine
Implements the damped fixed-point equilibrium solver and 1D/2D parameter sweep engines.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import copy
from .config import VehicleConfig, G_ACCEL, RAD_TO_DEG, DEG_TO_RAD
from .tire_model import TireModel, PacejkaNNTireModel, load_nn_tire_model
from .kinematics import KinematicsModel, VehicleKinematicsState
from .load_transfer import LoadTransferModel, CornerLoads, AxleLoadTransfer


@dataclass
class PointSolution:
    beta_rad: float
    delta_rad: float
    ay: float                 # [m/s^2]
    ay_g: float               # [g]
    mz: float                 # Total yaw moment [N*m]
    yaw_rate: float           # [rad/s]
    iterations: int
    converged: bool
    corner_loads: CornerLoads
    kinematics: VehicleKinematicsState
    fy_corners: Dict[str, float]  # [N]
    mz_corners: Dict[str, float]  # [N*m]


@dataclass
class YMDResult:
    """Complete results from a 2D YMD grid solve."""
    config: VehicleConfig
    beta_grid_rad: np.ndarray     # Shape (n_beta, n_delta)
    delta_grid_rad: np.ndarray    # Shape (n_beta, n_delta)
    beta_grid_deg: np.ndarray
    delta_grid_deg: np.ndarray
    ay_grid_ms2: np.ndarray       # Lateral acceleration [m/s^2]
    ay_grid_g: np.ndarray         # Lateral acceleration [g]
    mz_grid: np.ndarray           # Yaw moment [N*m]
    yaw_rate_grid: np.ndarray     # Yaw rate [rad/s]
    convergence_mask: np.ndarray  # Boolean mask of converged points
    point_solutions: List[List[PointSolution]]
    velocity_ms: float
    velocity_mph: float
    # High-resolution zero-angle isoline data for precise curves & KPI extraction
    beta0_hires_delta_deg: Optional[np.ndarray] = None
    beta0_hires_ay_g: Optional[np.ndarray] = None
    beta0_hires_mz: Optional[np.ndarray] = None
    delta0_hires_beta_deg: Optional[np.ndarray] = None
    delta0_hires_ay_g: Optional[np.ndarray] = None
    delta0_hires_mz: Optional[np.ndarray] = None


@dataclass
class ParameterSweepResult:
    """Results from a 1D or 2D parameter sweep."""
    param_names: List[str]
    param_values: List[np.ndarray]
    results: List[YMDResult]
    kpis: List[Any]  # List of VehicleKPIs


class YMDSolver:
    """
    Steady-state vehicle dynamics solver for generating Yaw Moment Diagrams.
    """

    def __init__(
        self,
        config: VehicleConfig,
        tire_model: Optional[TireModel] = None,
    ):
        self.config = config
        self.tire_model = tire_model if tire_model is not None else load_nn_tire_model()
        self.kinematics = KinematicsModel(config)
        self.load_transfer = LoadTransferModel(config)

    def solve_point(
        self,
        beta: float,
        delta: float,
        velocity: Optional[float] = None,
        initial_ay_guess: float = 0.0,
    ) -> PointSolution:
        """
        Solves the steady-state equilibrium at a single (beta, delta, velocity) point
        using damped fixed-point iteration on lateral acceleration.

        Parameters
        ----------
        beta : float
            Chassis slip angle in radians.
        delta : float
            Steering angle in radians.
        velocity : float, optional
            Forward speed in m/s (defaults to config simulation velocity).
        initial_ay_guess : float, optional
            Initial guess for lateral acceleration in m/s^2.
        """
        v = velocity if velocity is not None else self.config.simulation.velocity
        sim = self.config.simulation
        m = self.config.mass.total_mass
        a = self.config.a
        b = self.config.b
        tf = self.config.front_suspension.track_width
        tr = self.config.rear_suspension.track_width

        vx_cg = max(v * np.cos(beta), 0.1)

        ay_guess = initial_ay_guess
        converged = False
        iter_count = 0
        damping = sim.damping_factor
        tol = sim.ay_tolerance
        max_iter = sim.max_iterations

        # Solver loop
        for it in range(1, max_iter + 1):
            iter_count = it
            # 1. Steady state yaw rate: r = Ay / Vx
            yaw_rate = ay_guess / vx_cg

            # 2. 4-corner load transfer & roll angle
            corner_loads, axle_lt = self.load_transfer.compute_load_transfer(
                ay=ay_guess, velocity=v
            )

            # 3. 4-corner kinematics & slip angles
            kin = self.kinematics.compute_slip_angles(
                velocity=v,
                beta=beta,
                yaw_rate=yaw_rate,
                delta=delta,
                roll_angle=axle_lt.roll_angle,
            )

            # 4. Tire forces
            fy_fl, mz_fl = self.tire_model.compute_forces(
                fz=corner_loads.FL,
                alpha=kin.FL.slip_angle,
                camber=kin.FL.camber_angle,
                pressure=self.config.tire.inflation_pressure,
                velocity=v,
            )
            fy_fr, mz_fr = self.tire_model.compute_forces(
                fz=corner_loads.FR,
                alpha=kin.FR.slip_angle,
                camber=kin.FR.camber_angle,
                pressure=self.config.tire.inflation_pressure,
                velocity=v,
            )
            fy_rl, mz_rl = self.tire_model.compute_forces(
                fz=corner_loads.RL,
                alpha=kin.RL.slip_angle,
                camber=kin.RL.camber_angle,
                pressure=self.config.tire.inflation_pressure,
                velocity=v,
            )
            fy_rr, mz_rr = self.tire_model.compute_forces(
                fz=corner_loads.RR,
                alpha=kin.RR.slip_angle,
                camber=kin.RR.camber_angle,
                pressure=self.config.tire.inflation_pressure,
                velocity=v,
            )

            # 5. Resolve lateral forces into vehicle coordinate system
            # Note: tire forces Fy act perpendicular to wheel plane.
            # When steered by delta_wheel, body-lateral force is Fy * cos(delta)
            fy_fl_body = fy_fl * np.cos(kin.FL.steer_angle)
            fy_fr_body = fy_fr * np.cos(kin.FR.steer_angle)
            fy_rl_body = fy_rl * np.cos(kin.RL.steer_angle)
            fy_rr_body = fy_rr * np.cos(kin.RR.steer_angle)

            fy_total_body = fy_fl_body + fy_fr_body + fy_rl_body + fy_rr_body

            # 6. Compute new Ay
            ay_new = fy_total_body / m

            # 7. Check convergence
            if abs(ay_new - ay_guess) < tol:
                converged = True
                ay_guess = ay_new
                break

            # 8. Damped update
            ay_guess = (1.0 - damping) * ay_guess + damping * ay_new

        # Final yaw moment calculation around CG:
        # Front axle lever arm = +a, Rear axle lever arm = -b
        # Track width lever arms: left = +t/2, right = -t/2
        # Steered tire force creates body-x force = -Fy * sin(delta_wheel)
        fx_fl_body = -fy_fl * np.sin(kin.FL.steer_angle)
        fx_fr_body = -fy_fr * np.sin(kin.FR.steer_angle)
        fx_rl_body = -fy_rl * np.sin(kin.RL.steer_angle)
        fx_rr_body = -fy_rr * np.sin(kin.RR.steer_angle)

        mz_fl_arm = a * fy_fl_body - (+tf / 2.0) * fx_fl_body + mz_fl
        mz_fr_arm = a * fy_fr_body - (-tf / 2.0) * fx_fr_body + mz_fr
        mz_rl_arm = -b * fy_rl_body - (+tr / 2.0) * fx_rl_body + mz_rl
        mz_rr_arm = -b * fy_rr_body - (-tr / 2.0) * fx_rr_body + mz_rr

        mz_total = mz_fl_arm + mz_fr_arm + mz_rl_arm + mz_rr_arm

        fy_dict = {"FL": fy_fl, "FR": fy_fr, "RL": fy_rl, "RR": fy_rr}
        mz_dict = {"FL": mz_fl, "FR": mz_fr, "RL": mz_rl, "RR": mz_rr}

        return PointSolution(
            beta_rad=beta,
            delta_rad=delta,
            ay=ay_guess,
            ay_g=ay_guess / G_ACCEL,
            mz=mz_total,
            yaw_rate=ay_guess / vx_cg,
            iterations=iter_count,
            converged=converged,
            corner_loads=corner_loads,
            kinematics=kin,
            fy_corners=fy_dict,
            mz_corners=mz_dict,
        )

    def solve_grid(
        self,
        beta_sweep: Optional[np.ndarray] = None,
        delta_sweep: Optional[np.ndarray] = None,
        velocity: Optional[float] = None,
    ) -> YMDResult:
        """
        Solves the complete 2D YMD grid across chassis slip angle (beta) and steering angle (delta).
        Ensures exact zero lines (beta=0, delta=0) are included for precise KPI extraction.

        Parameters
        ----------
        beta_sweep : np.ndarray, optional
            Array of chassis slip angles in radians.
        delta_sweep : np.ndarray, optional
            Array of steering angles in radians.
        velocity : float, optional
            Forward speed in m/s.
        """
        sim = self.config.simulation
        v = velocity if velocity is not None else sim.velocity

        if beta_sweep is None:
            beta_vals = sim.beta_sweep.get_values()
        else:
            beta_vals = np.asarray(beta_sweep)

        if delta_sweep is None:
            delta_vals = sim.delta_sweep.get_values()
        else:
            delta_vals = np.asarray(delta_sweep)

        # Guarantee 0.0 is present in the sweep arrays for exact isoline tracking
        if not np.any(np.isclose(beta_vals, 0.0, atol=1e-5)):
            beta_vals = np.sort(np.unique(np.append(beta_vals, 0.0)))
        if not np.any(np.isclose(delta_vals, 0.0, atol=1e-5)):
            delta_vals = np.sort(np.unique(np.append(delta_vals, 0.0)))

        n_beta = len(beta_vals)
        n_delta = len(delta_vals)

        beta_grid, delta_grid = np.meshgrid(beta_vals, delta_vals, indexing="ij")

        ay_grid_ms2 = np.zeros((n_beta, n_delta))
        ay_grid_g = np.zeros((n_beta, n_delta))
        mz_grid = np.zeros((n_beta, n_delta))
        yaw_rate_grid = np.zeros((n_beta, n_delta))
        conv_mask = np.zeros((n_beta, n_delta), dtype=bool)

        solutions: List[List[PointSolution]] = []

        # Outer loop over Beta, inner loop over Delta
        for i, beta in enumerate(beta_vals):
            row_sol: List[PointSolution] = []
            ay_guess = 0.0  # Warm-start along steering sweep
            for j, delta in enumerate(delta_vals):
                sol = self.solve_point(
                    beta=beta,
                    delta=delta,
                    velocity=v,
                    initial_ay_guess=ay_guess,
                )
                ay_grid_ms2[i, j] = sol.ay
                ay_grid_g[i, j] = sol.ay_g
                mz_grid[i, j] = sol.mz
                yaw_rate_grid[i, j] = sol.yaw_rate
                conv_mask[i, j] = sol.converged
                row_sol.append(sol)
                # Warm-start guess for next delta point
                ay_guess = sol.ay
            solutions.append(row_sol)

        # High-resolution 1D sweeps for Beta=0 (varying Delta) and Delta=0 (varying Beta)
        d_min_rad = delta_vals.min()
        d_max_rad = delta_vals.max()
        b_min_rad = beta_vals.min()
        b_max_rad = beta_vals.max()

        hires_points = 81

        # 1. Beta = 0.0 sweep (varying Delta)
        d_hires = np.linspace(d_min_rad, d_max_rad, hires_points)
        b0_ay_g = np.zeros(hires_points)
        b0_mz = np.zeros(hires_points)
        ay_guess = 0.0
        for k, d_val in enumerate(d_hires):
            sol = self.solve_point(beta=0.0, delta=d_val, velocity=v, initial_ay_guess=ay_guess)
            b0_ay_g[k] = sol.ay_g
            b0_mz[k] = sol.mz
            ay_guess = sol.ay

        # 2. Delta = 0.0 sweep (varying Beta)
        b_hires = np.linspace(b_min_rad, b_max_rad, hires_points)
        d0_ay_g = np.zeros(hires_points)
        d0_mz = np.zeros(hires_points)
        ay_guess = 0.0
        for k, b_val in enumerate(b_hires):
            sol = self.solve_point(beta=b_val, delta=0.0, velocity=v, initial_ay_guess=ay_guess)
            d0_ay_g[k] = sol.ay_g
            d0_mz[k] = sol.mz
            ay_guess = sol.ay

        return YMDResult(
            config=self.config,
            beta_grid_rad=beta_grid,
            delta_grid_rad=delta_grid,
            beta_grid_deg=beta_grid * RAD_TO_DEG,
            delta_grid_deg=delta_grid * RAD_TO_DEG,
            ay_grid_ms2=ay_grid_ms2,
            ay_grid_g=ay_grid_g,
            mz_grid=mz_grid,
            yaw_rate_grid=yaw_rate_grid,
            convergence_mask=conv_mask,
            point_solutions=solutions,
            velocity_ms=v,
            velocity_mph=v * 2.23694,
            beta0_hires_delta_deg=d_hires * RAD_TO_DEG,
            beta0_hires_ay_g=b0_ay_g,
            beta0_hires_mz=b0_mz,
            delta0_hires_beta_deg=b_hires * RAD_TO_DEG,
            delta0_hires_ay_g=d0_ay_g,
            delta0_hires_mz=d0_mz,
        )

    def run_1d_sweep(
        self,
        setter_func: Callable[[VehicleConfig, float], None],
        values: np.ndarray,
        param_name: str = "Parameter",
    ) -> ParameterSweepResult:
        """
        Runs a 1D sweep over a vehicle setup parameter.

        Parameters
        ----------
        setter_func : Callable[[VehicleConfig, float], None]
            Function modifying a copy of VehicleConfig with the swept value.
        values : np.ndarray
            Array of parameter values to evaluate.
        param_name : str
            Display name of the parameter.
        """
        from .kpi import KPICalculator

        results = []
        kpis = []

        for val in values:
            cfg_copy = copy.deepcopy(self.config)
            setter_func(cfg_copy, val)
            solver_inst = YMDSolver(cfg_copy, tire_model=self.tire_model)
            res = solver_inst.solve_grid()
            kpi_inst = KPICalculator(res).compute_all_kpis(solver=solver_inst)
            results.append(res)
            kpis.append(kpi_inst)

        return ParameterSweepResult(
            param_names=[param_name],
            param_values=[values],
            results=results,
            kpis=kpis,
        )
