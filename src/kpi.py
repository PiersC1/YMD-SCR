"""
Vehicle Performance & Handling KPI Extraction Engine
Computes Steady State Grip Limit, Limit Balance, Control, Stability, and TLLTD.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from .solver import YMDResult
from .config import G_ACCEL, RAD_TO_DEG, DEG_TO_RAD


@dataclass
class VehicleKPIs:
    steady_state_grip_limit_g: float     # Maximum Ay at Mz = 0 [g]
    steady_state_grip_limit_ms2: float   # Maximum Ay at Mz = 0 [m/s^2]
    max_ay_g: float                      # Global peak Ay [g]
    max_ay_ms2: float                    # Global peak Ay [m/s^2]
    limit_balance_nm: float              # Yaw moment at max Ay [N*m] (< 0 = understeer, > 0 = oversteer)
    limit_balance_type: str              # "Understeer" / "Oversteer" / "Neutral"
    control_nm_per_deg: float            # dMz / dDelta at beta=0, delta=0 [N*m/deg]
    stability_nm_per_deg: float          # dMz / dBeta at delta=0, beta=0 [N*m/deg]
    tlltd_front_percent: float           # Front load transfer distribution [%]
    roll_gradient_deg_per_g: float       # Chassis roll angle per g [deg/g]
    velocity_mph: float


class KPICalculator:
    """
    Extracts standard vehicle dynamics and FSAE KPIs from solved YMD data grids.
    """

    def __init__(self, result: YMDResult):
        self.result = result

    def compute_steady_state_grip_limit(self) -> float:
        """
        Finds the maximum positive lateral acceleration where Yaw Moment Mz == 0.
        Uses zero-crossing interpolation along constant-steering and constant-slip curves.
        """
        ay_g = self.result.ay_grid_g
        mz = self.result.mz_grid

        # Collect all positive zero-crossings
        zero_crossing_ays = []

        # 1. Sweep across beta curves (rows)
        for i in range(mz.shape[0]):
            mz_row = mz[i, :]
            ay_row = ay_g[i, :]
            for j in range(len(mz_row) - 1):
                if (mz_row[j] <= 0 and mz_row[j + 1] >= 0) or (mz_row[j] >= 0 and mz_row[j + 1] <= 0):
                    if abs(mz_row[j + 1] - mz_row[j]) > 1e-4:
                        frac = -mz_row[j] / (mz_row[j + 1] - mz_row[j])
                        ay_cross = ay_row[j] + frac * (ay_row[j + 1] - ay_row[j])
                        if ay_cross > 0.0:
                            zero_crossing_ays.append(ay_cross)

        # 2. Sweep across delta curves (cols)
        for j in range(mz.shape[1]):
            mz_col = mz[:, j]
            ay_col = ay_g[:, j]
            for i in range(len(mz_col) - 1):
                if (mz_col[i] <= 0 and mz_col[i + 1] >= 0) or (mz_col[i] >= 0 and mz_col[i + 1] <= 0):
                    if abs(mz_col[i + 1] - mz_col[i]) > 1e-4:
                        frac = -mz_col[i] / (mz_col[i + 1] - mz_col[i])
                        ay_cross = ay_col[i] + frac * (ay_col[i + 1] - ay_col[i])
                        if ay_cross > 0.0:
                            zero_crossing_ays.append(ay_cross)

        if len(zero_crossing_ays) > 0:
            return float(np.max(zero_crossing_ays))

        # Fallback: find point closest to Mz=0 with positive Ay
        pos_mask = ay_g > 0
        if np.any(pos_mask):
            idx = np.argmin(np.abs(mz[pos_mask]))
            return float(ay_g[pos_mask][idx])
        return float(np.max(ay_g))

    def compute_limit_balance(self) -> Tuple[float, float, str]:
        """
        Finds the continuous global maximum positive lateral acceleration and the corresponding Mz
        using sub-grid 2D spline interpolation to prevent grid quantization jumping.

        Returns
        -------
        max_ay_g : float
        limit_balance_mz : float [N*m]
        balance_type : str ("Understeer" if Mz < 0, "Oversteer" if Mz > 0)
        """
        ay_g = self.result.ay_grid_g
        mz = self.result.mz_grid
        b_deg = self.result.beta_grid_deg[:, 0]
        d_deg = self.result.delta_grid_deg[0, :]

        try:
            from scipy.interpolate import RectBivariateSpline
            ay_spline = RectBivariateSpline(b_deg, d_deg, ay_g)
            mz_spline = RectBivariateSpline(b_deg, d_deg, mz)

            # Find discrete max seed
            max_idx = np.unravel_index(np.argmax(ay_g), ay_g.shape)
            b0, d0 = b_deg[max_idx[0]], d_deg[max_idx[1]]

            # Dense continuous local sub-grid around peak
            b_dense = np.linspace(max(b0 - 3.0, b_deg[0]), min(b0 + 3.0, b_deg[-1]), 80)
            d_dense = np.linspace(max(d0 - 3.0, d_deg[0]), min(d0 + 3.0, d_deg[-1]), 80)

            ay_dense = ay_spline(b_dense, d_dense)
            dense_idx = np.unravel_index(np.argmax(ay_dense), ay_dense.shape)

            b_opt = b_dense[dense_idx[0]]
            d_opt = d_dense[dense_idx[1]]
            max_ay = float(ay_dense[dense_idx])
            limit_mz = float(mz_spline(b_opt, d_opt)[0, 0])
        except Exception:
            # Fallback to discrete vertex
            max_idx = np.unravel_index(np.argmax(ay_g), ay_g.shape)
            max_ay = float(ay_g[max_idx])
            limit_mz = float(mz[max_idx])

        if limit_mz < -20.0:
            balance_type = "Understeer"
        elif limit_mz > 20.0:
            balance_type = "Oversteer"
        else:
            balance_type = "Neutral"

        return max_ay, limit_mz, balance_type

    def compute_control_authority(self, solver=None) -> float:
        """
        Calculates Control Authority = dMz / dDelta [N*m/deg] along beta = 0 at delta = 0.
        Uses high-precision central difference micro-solves if solver is provided,
        or dense high-resolution zero-angle isoline data if available.
        """
        if solver is not None:
            delta_eps = np.deg2rad(0.5)
            pt_pos = solver.solve_point(beta=0.0, delta=+delta_eps, velocity=self.result.velocity_ms)
            pt_neg = solver.solve_point(beta=0.0, delta=-delta_eps, velocity=self.result.velocity_ms)
            return float((pt_pos.mz - pt_neg.mz) / (2.0 * np.rad2deg(delta_eps)))

        if (
            self.result.beta0_hires_delta_deg is not None
            and self.result.beta0_hires_mz is not None
        ):
            d_vals = self.result.beta0_hires_delta_deg
            mz_vals = self.result.beta0_hires_mz
            d0_idx = np.argmin(np.abs(d_vals))
            if 0 < d0_idx < len(d_vals) - 1:
                return float((mz_vals[d0_idx + 1] - mz_vals[d0_idx - 1]) / (d_vals[d0_idx + 1] - d_vals[d0_idx - 1]))
            else:
                grad = np.gradient(mz_vals, d_vals)
                return float(grad[d0_idx])

        beta_vals = self.result.beta_grid_deg[:, 0]
        delta_vals = self.result.delta_grid_deg[0, :]

        # Find index of beta closest to 0
        beta_0_idx = np.argmin(np.abs(beta_vals))
        mz_along_beta0 = self.result.mz_grid[beta_0_idx, :]

        delta_0_idx = np.argmin(np.abs(delta_vals))
        if 0 < delta_0_idx < len(delta_vals) - 1:
            d_delta = delta_vals[delta_0_idx + 1] - delta_vals[delta_0_idx - 1]
            d_mz = mz_along_beta0[delta_0_idx + 1] - mz_along_beta0[delta_0_idx - 1]
            return float(d_mz / d_delta)
        else:
            grad = np.gradient(mz_along_beta0, delta_vals)
            return float(grad[delta_0_idx])

    def compute_stability(self, solver=None) -> float:
        """
        Calculates Yaw Stability = dMz / dBeta [N*m/deg] along delta = 0 at beta = 0.
        Uses high-precision central difference micro-solves if solver is provided,
        or dense high-resolution zero-angle isoline data if available.
        """
        if solver is not None:
            beta_eps = np.deg2rad(0.5)
            pt_pos = solver.solve_point(beta=+beta_eps, delta=0.0, velocity=self.result.velocity_ms)
            pt_neg = solver.solve_point(beta=-beta_eps, delta=0.0, velocity=self.result.velocity_ms)
            return float((pt_pos.mz - pt_neg.mz) / (2.0 * np.rad2deg(beta_eps)))

        if (
            self.result.delta0_hires_beta_deg is not None
            and self.result.delta0_hires_mz is not None
        ):
            b_vals = self.result.delta0_hires_beta_deg
            mz_vals = self.result.delta0_hires_mz
            b0_idx = np.argmin(np.abs(b_vals))
            if 0 < b0_idx < len(b_vals) - 1:
                return float((mz_vals[b0_idx + 1] - mz_vals[b0_idx - 1]) / (b_vals[b0_idx + 1] - b_vals[b0_idx - 1]))
            else:
                grad = np.gradient(mz_vals, b_vals)
                return float(grad[b0_idx])

        beta_vals = self.result.beta_grid_deg[:, 0]
        delta_vals = self.result.delta_grid_deg[0, :]

        delta_0_idx = np.argmin(np.abs(delta_vals))
        mz_along_delta0 = self.result.mz_grid[:, delta_0_idx]

        beta_0_idx = np.argmin(np.abs(beta_vals))
        if 0 < beta_0_idx < len(beta_vals) - 1:
            d_beta = beta_vals[beta_0_idx + 1] - beta_vals[beta_0_idx - 1]
            d_mz = mz_along_delta0[beta_0_idx + 1] - mz_along_delta0[beta_0_idx - 1]
            return float(d_mz / d_beta)
        else:
            grad = np.gradient(mz_along_delta0, beta_vals)
            return float(grad[beta_0_idx])

    def compute_all_kpis(self, solver=None) -> VehicleKPIs:
        """
        Computes and returns the full set of handling and stability KPIs.
        Optionally accepts solver instance for targeted local adaptive refinement.
        """
        grip_limit_g = self.compute_steady_state_grip_limit()
        grip_limit_ms2 = grip_limit_g * G_ACCEL

        max_ay_g, limit_balance_nm, balance_type = self.compute_limit_balance()
        max_ay_ms2 = max_ay_g * G_ACCEL

        control = self.compute_control_authority(solver=solver)
        stability = self.compute_stability(solver=solver)

        # TLLTD and roll gradient from vehicle load transfer model
        from .load_transfer import LoadTransferModel
        lt = LoadTransferModel(self.result.config)
        _, axle_lt_1g = lt.compute_load_transfer(ay=G_ACCEL, velocity=self.result.velocity_ms)
        tlltd = axle_lt_1g.tlltd
        roll_grad_deg_per_g = axle_lt_1g.roll_angle * RAD_TO_DEG

        return VehicleKPIs(
            steady_state_grip_limit_g=round(grip_limit_g, 3),
            steady_state_grip_limit_ms2=round(grip_limit_ms2, 2),
            max_ay_g=round(max_ay_g, 3),
            max_ay_ms2=round(max_ay_ms2, 2),
            limit_balance_nm=round(limit_balance_nm, 1),
            limit_balance_type=balance_type,
            control_nm_per_deg=round(control, 2),
            stability_nm_per_deg=round(stability, 2),
            tlltd_front_percent=round(tlltd, 1),
            roll_gradient_deg_per_g=round(roll_grad_deg_per_g, 2),
            velocity_mph=round(self.result.velocity_mph, 1),
        )
