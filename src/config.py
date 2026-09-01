"""
Vehicle Configuration & Unit Conversion Module
Parses vehicle YAML configuration files and standardizes all parameters to SI units.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import numpy as np

# Standard SI conversion constants
INCH_TO_M = 0.0254
M_TO_INCH = 1.0 / INCH_TO_M
LB_TO_KG = 0.45359237
KG_TO_LB = 1.0 / LB_TO_KG
LBF_TO_N = 4.4482216152605
N_TO_LBF = 1.0 / LBF_TO_N
LBF_IN_TO_NM = 0.112984829
NM_TO_LBF_IN = 1.0 / LBF_IN_TO_NM
LBF_PER_IN_TO_N_PER_M = 175.126835
N_PER_M_TO_LBF_PER_IN = 1.0 / LBF_PER_IN_TO_N_PER_M
LB_PER_FT3_TO_KG_PER_M3 = 16.018463
MPH_TO_MS = 0.44704
MS_TO_MPH = 1.0 / MPH_TO_MS
PSI_TO_PA = 6894.75729
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
G_ACCEL = 9.80665


def _extract_val(entry: Union[Dict[str, Any], float, int], default_unit: str = "") -> float:
    """Helper to extract float value from either a plain number or a dict with value/unit."""
    if isinstance(entry, dict):
        return float(entry.get("value", 0.0))
    return float(entry)


@dataclass
class AeroParams:
    Cl: float = 4.29               # Total vehicle lift/downforce coefficient (positive = downforce)
    CoP: float = 45.37             # Center of Pressure (% front, 0 to 100)
    rho: float = 1.225             # Air density in kg/m^3 (converted from lb/ft^3 if specified)
    frontal_area: float = 1.14     # Frontal area in m^2

    @property
    def cop_fraction_front(self) -> float:
        return self.CoP / 100.0


@dataclass
class MassParams:
    total_mass: float = 294.8      # Total vehicle + driver + fuel mass in kg
    dry_mass: float = 222.26       # Dry mass in kg
    driver_mass: float = 68.04     # Driver mass in kg
    fuel_mass: float = 4.54        # Fuel mass in kg
    x_loc_front: float = 0.49      # Longitudinal CG position (fraction from front axle: 0.49 means 49% front)
    y_loc_left: float = 0.501      # Lateral CG position (fraction from left: 0.501 means 50.1% left)
    cg_height: float = 0.3175      # CG height in meters (z_loc)


@dataclass
class Dimensions:
    wheelbase: float = 1.5367      # Wheelbase in meters (L)

    @property
    def a(self) -> float:
        """Distance from CG to front axle (m) - computed with x_loc_front."""
        return self.wheelbase * (1.0 - 0.49)  # placeholder, vehicle model uses actual x_loc

    @property
    def b(self) -> float:
        """Distance from CG to rear axle (m) - computed with x_loc_front."""
        return self.wheelbase * 0.49


@dataclass
class SuspensionAxleParams:
    spring_rate: float = 52538.0       # Wheel/spring stiffness in N/m (converted from lb/in)
    arb_stiffness: float = 0.0         # Anti-roll bar stiffness in N/m (converted from lb/in)
    track_width: float = 1.2192        # Track width in meters
    static_camber: float = 0.0349      # Static Inclination Angle / Camber in radians
    roll_center_height: float = 0.0691 # Roll center height in meters
    static_toe: float = -0.0087        # Static toe angle in radians (toe-out is negative, toe-in is positive)
    spring_MR: float = 1.08            # Spring motion ratio (wheel displacement / spring displacement)
    arb_MR: float = 0.0                # ARB motion ratio
    unsprung_mass: float = 4.90        # Unsprung mass per axle corner in kg (or per axle)
    unsprung_cg_height: float = 0.2032 # Unsprung CG height in meters


@dataclass
class SteeringParams:
    ackermann_percent: float = 110.0   # Ackermann percentage (100% = true geometric Ackermann)


@dataclass
class TireConfigParams:
    inflation_pressure: float = 75842.0 # Inflation pressure in Pa (e.g. 11 psi)
    tire_stiffness: float = 78807.0     # Radial tire stiffness in N/m (e.g. 450 lb/in)
    model_weights_path: str = "NN_model/R20_tire_model.pth"
    norm_params_path: str = "NN_model/norm_params.json"


@dataclass
class SweepConfig:
    min_val: float
    max_val: float
    points: int
    unit: str

    def get_values(self) -> np.ndarray:
        """Return array of sweep values in SI units (rad for deg, etc.)."""
        vals = np.linspace(self.min_val, self.max_val, self.points)
        if self.unit.lower() in ["deg", "degree", "degrees"]:
            return vals * DEG_TO_RAD
        elif self.unit.lower() in ["mph"]:
            return vals * MPH_TO_MS
        return vals

    def get_display_values(self) -> np.ndarray:
        """Return array of sweep values in user display units."""
        return np.linspace(self.min_val, self.max_val, self.points)


@dataclass
class SimulationParams:
    velocity: float = 11.176           # Velocity in m/s (default 25 mph)
    velocity_display: float = 25.0     # Velocity in display unit (mph)
    velocity_unit: str = "mph"
    beta_sweep: SweepConfig = field(default_factory=lambda: SweepConfig(-15.0, 15.0, 15, "deg"))
    delta_sweep: SweepConfig = field(default_factory=lambda: SweepConfig(-20.0, 20.0, 21, "deg"))
    slip_ratio_sweep: SweepConfig = field(default_factory=lambda: SweepConfig(0.0, 0.0, 1, "none"))
    ay_tolerance: float = 0.005        # Ay convergence threshold in m/s^2
    max_iterations: int = 150          # Max inner loop solver iterations
    damping_factor: float = 0.35       # Damping relaxation factor (lambda: 0.35 means 65% old + 35% new)


@dataclass
class VehicleConfig:
    aero: AeroParams = field(default_factory=AeroParams)
    mass: MassParams = field(default_factory=MassParams)
    dimensions: Dimensions = field(default_factory=Dimensions)
    front_suspension: SuspensionAxleParams = field(default_factory=SuspensionAxleParams)
    rear_suspension: SuspensionAxleParams = field(default_factory=SuspensionAxleParams)
    steering: SteeringParams = field(default_factory=SteeringParams)
    tire: TireConfigParams = field(default_factory=TireConfigParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @property
    def a(self) -> float:
        """Distance from CG to front axle in meters."""
        return self.dimensions.wheelbase * (1.0 - self.mass.x_loc_front)

    @property
    def b(self) -> float:
        """Distance from CG to rear axle in meters."""
        return self.dimensions.wheelbase * self.mass.x_loc_front


def load_config(yaml_path: Union[str, Path]) -> VehicleConfig:
    """
    Loads and parses a vehicle YAML configuration file, converting all units to SI.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # 1. Aero params
    aero_data = data.get("aero_params", {})
    cl = float(aero_data.get("Cl", 4.29))
    cop = float(aero_data.get("CoP", 45.37))
    
    rho_entry = aero_data.get("rho", 0.0763)
    if isinstance(rho_entry, dict):
        rho_val = float(rho_entry.get("value", 0.0763))
        rho_unit = rho_entry.get("unit", "lb/ft^3")
        rho = rho_val * LB_PER_FT3_TO_KG_PER_M3 if "lb" in rho_unit else rho_val
    else:
        rho = float(rho_entry) * LB_PER_FT3_TO_KG_PER_M3

    area_entry = aero_data.get("frontal_area", 1.14)
    frontal_area = _extract_val(area_entry)

    aero = AeroParams(Cl=cl, CoP=cop, rho=rho, frontal_area=frontal_area)

    # 2. Mass params
    mass_data = data.get("mass", {})
    x_loc = float(mass_data.get("x_loc", 0.49))
    y_loc = float(mass_data.get("y_loc", 0.501))
    
    z_loc_entry = mass_data.get("z_loc", 12.5)
    z_loc_val = _extract_val(z_loc_entry)
    cg_height = z_loc_val * INCH_TO_M

    dry_mass = _extract_val(mass_data.get("dry_mass", 490.0)) * LB_TO_KG
    driver_mass = _extract_val(mass_data.get("driver_mass", 150.0)) * LB_TO_KG
    fuel_mass = _extract_val(mass_data.get("fuel_mass", 10.0)) * LB_TO_KG
    total_mass = dry_mass + driver_mass + fuel_mass

    mass = MassParams(
        total_mass=total_mass,
        dry_mass=dry_mass,
        driver_mass=driver_mass,
        fuel_mass=fuel_mass,
        x_loc_front=x_loc,
        y_loc_left=y_loc,
        cg_height=cg_height,
    )

    # 3. Dimensions
    dims_data = data.get("dimensions", {})
    wb_val = _extract_val(dims_data.get("wheelbase", 60.5)) * INCH_TO_M
    dimensions = Dimensions(wheelbase=wb_val)

    # 4. Front Suspension
    fs_data = data.get("frontSuspension", {})
    fs_stiff = fs_data.get("stiffness", {})
    f_k_spring = _extract_val(fs_stiff.get("spring_rate", 300.0)) * LBF_PER_IN_TO_N_PER_M
    f_k_arb = _extract_val(fs_stiff.get("arb_stiffness", 0.0)) * LBF_PER_IN_TO_N_PER_M

    fs_geom = fs_data.get("geom", {})
    f_tw = _extract_val(fs_geom.get("track_width", 48.0)) * INCH_TO_M
    f_ia = _extract_val(fs_geom.get("static_IA", 2.0)) * DEG_TO_RAD
    f_rc = _extract_val(fs_geom.get("static_roll_center", 2.72)) * INCH_TO_M
    f_toe = _extract_val(fs_geom.get("static_toe", -0.5)) * DEG_TO_RAD

    fs_kin = fs_data.get("kinematics", {})
    f_smr = float(fs_kin.get("spring_MR", 1.08))
    f_amr = float(fs_kin.get("arb_MR", 0.0))

    fs_mass = fs_data.get("mass", {})
    f_umass = _extract_val(fs_mass.get("unsprung_mass", 10.8)) * LB_TO_KG
    f_ucg = _extract_val(fs_mass.get("cg_height", 8.0)) * INCH_TO_M

    front_suspension = SuspensionAxleParams(
        spring_rate=f_k_spring,
        arb_stiffness=f_k_arb,
        track_width=f_tw,
        static_camber=f_ia,
        roll_center_height=f_rc,
        static_toe=f_toe,
        spring_MR=f_smr,
        arb_MR=f_amr,
        unsprung_mass=f_umass,
        unsprung_cg_height=f_ucg,
    )

    # 5. Rear Suspension
    rs_data = data.get("rearSuspension", {})
    rs_stiff = rs_data.get("stiffness", {})
    r_k_spring = _extract_val(rs_stiff.get("spring_rate", 250.0)) * LBF_PER_IN_TO_N_PER_M
    r_k_arb = _extract_val(rs_stiff.get("arb_stiffness", 0.0)) * LBF_PER_IN_TO_N_PER_M

    rs_geom = rs_data.get("geom", {})
    r_tw = _extract_val(rs_geom.get("track_width", 48.0)) * INCH_TO_M
    r_ia = _extract_val(rs_geom.get("static_IA", 0.5)) * DEG_TO_RAD
    r_rc = _extract_val(rs_geom.get("static_roll_center", 4.75)) * INCH_TO_M
    r_toe = _extract_val(rs_geom.get("static_toe", 0.0)) * DEG_TO_RAD

    rs_kin = rs_data.get("kinematics", {})
    r_smr = float(rs_kin.get("spring_MR", 1.18))
    r_amr = float(rs_kin.get("arb_MR", 0.0))

    rs_mass = rs_data.get("mass", {})
    r_umass = _extract_val(rs_mass.get("unsprung_mass", 12.43)) * LB_TO_KG
    r_ucg = _extract_val(rs_mass.get("cg_height", 8.0)) * INCH_TO_M

    rear_suspension = SuspensionAxleParams(
        spring_rate=r_k_spring,
        arb_stiffness=r_k_arb,
        track_width=r_tw,
        static_camber=r_ia,
        roll_center_height=r_rc,
        static_toe=r_toe,
        spring_MR=r_smr,
        arb_MR=r_amr,
        unsprung_mass=r_umass,
        unsprung_cg_height=r_ucg,
    )

    # 6. Steering
    steer_data = data.get("steering", {})
    ack_entry = steer_data.get("ackermann", 110.0)
    ack_val = _extract_val(ack_entry)
    steering = SteeringParams(ackermann_percent=ack_val)

    # 7. Tires
    tire_data = data.get("tire_params", {})
    infl_pres = _extract_val(tire_data.get("inflation_pres", 11.0)) * PSI_TO_PA
    tire_k = _extract_val(tire_data.get("tire_stiffness", 450.0)) * LBF_PER_IN_TO_N_PER_M
    tires = TireConfigParams(
        inflation_pressure=infl_pres,
        tire_stiffness=tire_k,
    )

    # 8. Simulation params
    sim_data = data.get("simulation_params", {}).get("YMD", {})
    vel_entry = sim_data.get("velocity", 25.0)
    vel_disp = _extract_val(vel_entry)
    vel_unit = vel_entry.get("unit", "mph") if isinstance(vel_entry, dict) else "mph"
    vel_ms = vel_disp * MPH_TO_MS if "mph" in vel_unit.lower() else vel_disp

    sweeps_data = sim_data.get("sweeps", {})
    
    sa_swp = sweeps_data.get("slip_angle", {"min": -15, "max": 15, "points": 15, "unit": "deg"})
    beta_sweep = SweepConfig(
        min_val=float(sa_swp.get("min", -15.0)),
        max_val=float(sa_swp.get("max", 15.0)),
        points=int(sa_swp.get("points", 15)),
        unit=str(sa_swp.get("unit", "deg")),
    )

    st_swp = sweeps_data.get("steering_angle", {"min": -20, "max": 20, "points": 21, "unit": "deg"})
    delta_sweep = SweepConfig(
        min_val=float(st_swp.get("min", -20.0)),
        max_val=float(st_swp.get("max", 20.0)),
        points=int(st_swp.get("points", 21)),
        unit=str(st_swp.get("unit", "deg")),
    )

    sr_swp = sweeps_data.get("slip_ratio", {"min": 0, "max": 0, "points": 1, "unit": "none"})
    slip_ratio_sweep = SweepConfig(
        min_val=float(sr_swp.get("min", 0.0)),
        max_val=float(sr_swp.get("max", 0.0)),
        points=int(sr_swp.get("points", 1)),
        unit=str(sr_swp.get("unit", "none")),
    )

    simulation = SimulationParams(
        velocity=vel_ms,
        velocity_display=vel_disp,
        velocity_unit=vel_unit,
        beta_sweep=beta_sweep,
        delta_sweep=delta_sweep,
        slip_ratio_sweep=slip_ratio_sweep,
    )

    return VehicleConfig(
        aero=aero,
        mass=mass,
        dimensions=dimensions,
        front_suspension=front_suspension,
        rear_suspension=rear_suspension,
        steering=steering,
        tire=tires,
        simulation=simulation,
        raw_dict=data,
    )
