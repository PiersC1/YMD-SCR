"""
Vehicle Kinematics & Ackermann Steering Module
Calculates 4-corner velocities, steer angles, and tire slip angles.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .config import VehicleConfig


@dataclass
class WheelKinematics:
    """Individual tire kinematics state."""
    slip_angle: float       # alpha [rad]
    steer_angle: float      # delta_wheel [rad] (front wheels steered, rear wheels usually 0)
    camber_angle: float     # gamma [rad]
    vx_corner: float        # local forward velocity [m/s]
    vy_corner: float        # local lateral velocity [m/s]


@dataclass
class VehicleKinematicsState:
    """Complete 4-corner kinematics state."""
    FL: WheelKinematics
    FR: WheelKinematics
    RL: WheelKinematics
    RR: WheelKinematics
    yaw_rate: float         # r [rad/s]
    vx_cg: float            # Vx at CG [m/s]
    vy_cg: float            # Vy at CG [m/s]


class KinematicsModel:
    """
    Computes corner velocities, Ackermann steer distribution, and tire slip angles.
    """

    def __init__(self, config: VehicleConfig):
        self.config = config
        self.L = config.dimensions.wheelbase
        self.a = config.a  # CG to front axle (m)
        self.b = config.b  # CG to rear axle (m)
        self.tf = config.front_suspension.track_width
        self.tr = config.rear_suspension.track_width
        self.ackermann_pct = config.steering.ackermann_percent

    def compute_steer_angles(self, delta_handwheel: float) -> Tuple[float, float]:
        """
        Compute left and right front wheel steer angles based on Ackermann percentage.

        Parameters
        ----------
        delta_handwheel : float
            Effective road wheel steer angle (average / nominal) in radians.
            Positive = left turn.

        Returns
        -------
        delta_fl, delta_fr : Tuple[float, float]
            Steer angles for Front-Left and Front-Right wheels in radians.
        """
        if abs(delta_handwheel) < 1e-6:
            return 0.0, 0.0

        # Sign of turn (positive = left turn)
        is_left = delta_handwheel > 0
        delta_abs = abs(delta_handwheel)

        # 100% geometric Ackermann:
        # cot(delta_outer) - cot(delta_inner) = t_f / L
        # For a left turn: inner is Left (FL), outer is Right (FR)
        # 1 / tan(delta_inner) = 1 / tan(delta) - t_f / (2 * L)
        # 1 / tan(delta_outer) = 1 / tan(delta) + t_f / (2 * L)
        cot_delta = 1.0 / np.tan(delta_abs)
        t_over_2L = self.tf / (2.0 * self.L)

        cot_inner = cot_delta - t_over_2L
        cot_outer = cot_delta + t_over_2L

        delta_inner_100 = np.arctan(1.0 / max(cot_inner, 1e-4))
        delta_outer_100 = np.arctan(1.0 / max(cot_outer, 1e-4))

        # Parallel steering: delta_inner = delta_outer = delta_abs
        ack_frac = self.ackermann_pct / 100.0

        delta_inner = (1.0 - ack_frac) * delta_abs + ack_frac * delta_inner_100
        delta_outer = (1.0 - ack_frac) * delta_abs + ack_frac * delta_outer_100

        if is_left:
            delta_fl = delta_inner
            delta_fr = delta_outer
        else:
            delta_fl = -delta_outer
            delta_fr = -delta_inner

        return delta_fl, delta_fr

    def compute_slip_angles(
        self,
        velocity: float,
        beta: float,
        yaw_rate: float,
        delta: float,
        roll_angle: float = 0.0,
    ) -> VehicleKinematicsState:
        """
        Calculates 4-corner velocities and tire slip angles.

        Parameters
        ----------
        velocity : float
            Vehicle total speed at CG in m/s.
        beta : float
            Chassis slip angle in radians.
        yaw_rate : float
            Yaw rate in rad/s.
        delta : float
            Nominal steering angle in radians.
        roll_angle : float
            Chassis roll angle in radians.
        """
        vx_cg = velocity * np.cos(beta)
        vy_cg = velocity * np.sin(beta)

        # Steer angles
        delta_fl, delta_fr = self.compute_steer_angles(delta)
        delta_rl, delta_rr = 0.0, 0.0

        # Corner positions relative to CG: (x forward, y left)
        # FL: (+a, +tf/2)
        # FR: (+a, -tf/2)
        # RL: (-b, +tr/2)
        # RR: (-b, -tr/2)
        r = yaw_rate

        # Velocities in vehicle body frame at each wheel center:
        # vx_i = Vx - r * y_i
        # vy_i = Vy + r * x_i
        vx_fl = vx_cg - r * (+self.tf / 2.0)
        vy_fl = vy_cg + r * (+self.a)

        vx_fr = vx_cg - r * (-self.tf / 2.0)
        vy_fr = vy_cg + r * (+self.a)

        vx_rl = vx_cg - r * (+self.tr / 2.0)
        vy_rl = vy_cg + r * (-self.b)

        vx_rr = vx_cg - r * (-self.tr / 2.0)
        vy_rr = vy_cg + r * (-self.b)

        # Static alignments
        toe_f = self.config.front_suspension.static_toe
        toe_r = self.config.rear_suspension.static_toe
        camber_f0 = self.config.front_suspension.static_camber
        camber_r0 = self.config.rear_suspension.static_camber

        # Slip angle definition: alpha = arctan2(vy_wheel, vx_wheel) - delta_wheel_total
        # Where delta_wheel_total = delta + toe
        # Left toe > 0 is toe-in (turns right -> subtracts from slip), toe < 0 is toe-out
        alpha_fl = np.arctan2(vy_fl, vx_fl) - (delta_fl + toe_f)
        alpha_fr = np.arctan2(vy_fr, vx_fr) - (delta_fr - toe_f)
        alpha_rl = np.arctan2(vy_rl, vx_rl) - (delta_rl + toe_r)
        alpha_rr = np.arctan2(vy_rr, vx_rr) - (delta_rr - toe_r)

        # Dynamic camber under chassis roll:
        camber_fl = camber_f0 - roll_angle
        camber_fr = camber_f0 + roll_angle
        camber_rl = camber_r0 - roll_angle
        camber_rr = camber_r0 + roll_angle

        fl = WheelKinematics(alpha_fl, delta_fl, camber_fl, vx_fl, vy_fl)
        fr = WheelKinematics(alpha_fr, delta_fr, camber_fr, vx_fr, vy_fr)
        rl = WheelKinematics(alpha_rl, delta_rl, camber_rl, vx_rl, vy_rl)
        rr = WheelKinematics(alpha_rr, delta_rr, camber_rr, vx_rr, vy_rr)

        return VehicleKinematicsState(
            FL=fl, FR=fr, RL=rl, RR=rr, yaw_rate=r, vx_cg=vx_cg, vy_cg=vy_cg
        )
