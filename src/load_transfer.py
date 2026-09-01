"""
4-Corner Quasi-Static Roll & Lateral Load Transfer Model
Accounts for sprung roll moment, roll center heights, ARB/spring stiffness,
unsprung masses, and aerodynamic downforce distribution.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .config import VehicleConfig, G_ACCEL


@dataclass
class AxleLoadTransfer:
    delta_fz_front: float  # [N]
    delta_fz_rear: float   # [N]
    tlltd: float           # Front load transfer distribution [%]
    roll_angle: float      # Chassis roll angle [rad]


@dataclass
class CornerLoads:
    FL: float  # [N]
    FR: float  # [N]
    RL: float  # [N]
    RR: float  # [N]
    aero_downforce: float  # [N]


class LoadTransferModel:
    """
    Computes static loads, roll stiffnesses, roll angles, lateral weight transfer,
    and 4-corner normal forces.
    """

    def __init__(self, config: VehicleConfig):
        self.config = config
        self.m = config.mass.total_mass
        self.L = config.dimensions.wheelbase
        self.a = config.a
        self.b = config.b
        self.tf = config.front_suspension.track_width
        self.tr = config.rear_suspension.track_width
        self.cg_h = config.mass.cg_height

        # Unsprung masses (2 wheels per axle)
        self.m_uf = config.front_suspension.unsprung_mass * 2.0
        self.m_ur = config.rear_suspension.unsprung_mass * 2.0
        self.m_s = max(self.m - (self.m_uf + self.m_ur), self.m * 0.7)

        # Roll center heights
        self.h_rc_f = config.front_suspension.roll_center_height
        self.h_rc_r = config.rear_suspension.roll_center_height
        self.h_rc_cg = (self.h_rc_f * self.b + self.h_rc_r * self.a) / self.L
        self.h_arm = max(self.cg_h - self.h_rc_cg, 0.05)  # roll moment arm

        # Unsprung CG heights
        self.h_uf = config.front_suspension.unsprung_cg_height
        self.h_ur = config.rear_suspension.unsprung_cg_height

        # Roll stiffness calculations:
        # Spring roll stiffness: 0.5 * k_spring * (MR_spring)^2 * (track)^2  [N*m/rad]
        fs = config.front_suspension
        rs = config.rear_suspension

        self.k_phi_spring_f = 0.5 * fs.spring_rate * (fs.spring_MR ** 2) * (self.tf ** 2)
        self.k_phi_arb_f = fs.arb_stiffness * (fs.arb_MR ** 2) * (self.tf ** 2)
        self.k_phi_f = self.k_phi_spring_f + self.k_phi_arb_f

        self.k_phi_spring_r = 0.5 * rs.spring_rate * (rs.spring_MR ** 2) * (self.tr ** 2)
        self.k_phi_arb_r = rs.arb_stiffness * (rs.arb_MR ** 2) * (self.tr ** 2)
        self.k_phi_r = self.k_phi_spring_r + self.k_phi_arb_r

        self.k_phi_total = max(self.k_phi_f + self.k_phi_r, 1000.0)

        # Static loads
        total_weight = self.m * G_ACCEL
        x_frac_f = config.mass.x_loc_front
        y_frac_l = config.mass.y_loc_left

        self.fz_static_front = total_weight * x_frac_f
        self.fz_static_rear = total_weight * (1.0 - x_frac_f)

        self.fz_static_fl = self.fz_static_front * y_frac_l
        self.fz_static_fr = self.fz_static_front * (1.0 - y_frac_l)
        self.fz_static_rl = self.fz_static_rear * y_frac_l
        self.fz_static_rr = self.fz_static_rear * (1.0 - y_frac_l)

    def compute_aero_loads(self, velocity: float) -> Tuple[float, float, float]:
        """
        Calculates aerodynamic downforce and distribution.

        Returns
        -------
        fz_aero_front : float [N]
        fz_aero_rear : float [N]
        total_downforce : float [N]
        """
        aero = self.config.aero
        # Aero Downforce = 0.5 * rho * A * Cl * V^2
        q = 0.5 * aero.rho * aero.frontal_area * aero.Cl * (velocity ** 2)
        cop_f = aero.cop_fraction_front
        fz_aero_f = q * cop_f
        fz_aero_r = q * (1.0 - cop_f)
        return fz_aero_f, fz_aero_r, q

    def compute_load_transfer(self, ay: float, velocity: float) -> Tuple[CornerLoads, AxleLoadTransfer]:
        """
        Compute 4-corner normal loads and axle load transfers for a given lateral acceleration.

        Parameters
        ----------
        ay : float
            Lateral acceleration in m/s^2 (positive = left turn / accelerating towards left).
        velocity : float
            Vehicle forward speed in m/s.
        """
        # 1. Roll angle: phi = (m_s * ay * h_arm) / (K_phi_total - m_s * g * h_arm)
        roll_denom = self.k_phi_total - self.m_s * G_ACCEL * self.h_arm
        if roll_denom <= 100.0:
            roll_denom = 100.0
        roll_angle = (self.m_s * ay * self.h_arm) / roll_denom

        # 2. Axle lateral load transfers:
        # Delta Fz = (1 / track) * [ m_u * ay * h_u + m_s * ay * (dist/L) * h_rc + K_phi * phi ]
        # For left turn (ay > 0), load transfers to the right wheels (+FR, +RR, -FL, -RL)
        delta_fz_f = (1.0 / self.tf) * (
            self.m_uf * ay * self.h_uf
            + self.m_s * ay * (self.b / self.L) * self.h_rc_f
            + self.k_phi_f * roll_angle
        )

        delta_fz_r = (1.0 / self.tr) * (
            self.m_ur * ay * self.h_ur
            + self.m_s * ay * (self.a / self.L) * self.h_rc_r
            + self.k_phi_r * roll_angle
        )

        total_d_fz = abs(delta_fz_f) + abs(delta_fz_r)
        tlltd = (abs(delta_fz_f) / total_d_fz * 100.0) if total_d_fz > 1e-4 else (self.k_phi_f / self.k_phi_total * 100.0)

        # 3. Aero downforce
        fz_aero_f, fz_aero_r, total_aero = self.compute_aero_loads(velocity)

        # 4. Instantaneous normal load per corner
        fz_fl = max(0.0, self.fz_static_fl + (fz_aero_f / 2.0) - delta_fz_f)
        fz_fr = max(0.0, self.fz_static_fr + (fz_aero_f / 2.0) + delta_fz_f)
        fz_rl = max(0.0, self.fz_static_rl + (fz_aero_r / 2.0) - delta_fz_r)
        fz_rr = max(0.0, self.fz_static_rr + (fz_aero_r / 2.0) + delta_fz_r)

        corner_loads = CornerLoads(
            FL=fz_fl,
            FR=fz_fr,
            RL=fz_rl,
            RR=fz_rr,
            aero_downforce=total_aero,
        )

        axle_lt = AxleLoadTransfer(
            delta_fz_front=delta_fz_f,
            delta_fz_rear=delta_fz_r,
            tlltd=tlltd,
            roll_angle=roll_angle,
        )

        return corner_loads, axle_lt
