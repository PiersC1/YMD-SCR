"""
Modular Tire Model Interface & Neural Network / Pacejka Wrapper
Standardizes all input/output quantities to SI units (N, N*m, radians, Pa, m/s).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union, Optional
import json
import numpy as np
import torch
import torch.nn as nn


class TireModel(ABC):
    """Abstract base class for all tire models in the YMD solver."""

    @abstractmethod
    def compute_forces(
        self,
        fz: float,
        alpha: float,
        camber: float = 0.0,
        pressure: float = 75842.0,
        velocity: float = 11.176,
    ) -> Tuple[float, float]:
        """
        Compute lateral force (Fy) and self-aligning torque (Mz).

        Parameters
        ----------
        fz : float
            Positive normal load (compressive load on tire) in Newtons [N]. Must be >= 0.
        alpha : float
            Tire slip angle in radians [rad].
        camber : float
            Inclination / camber angle in radians [rad].
        pressure : float
            Tire inflation pressure in Pascals [Pa].
        velocity : float
            Tire forward velocity in meters per second [m/s].

        Returns
        -------
        fy : float
            Lateral force in Newtons [N].
        mz : float
            Self-aligning torque in Newton-meters [N*m].
        """
        pass

    def compute_forces_batch(
        self,
        fz_arr: np.ndarray,
        alpha_arr: np.ndarray,
        camber_arr: Optional[np.ndarray] = None,
        pressure: float = 75842.0,
        velocity: float = 11.176,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized computation of lateral force (Fy) and aligning torque (Mz)."""
        fz_flat = np.asarray(fz_arr).flatten()
        alpha_flat = np.asarray(alpha_arr).flatten()
        n = len(fz_flat)
        camber_flat = np.zeros(n) if camber_arr is None else np.asarray(camber_arr).flatten()

        fy_list = np.zeros(n)
        mz_list = np.zeros(n)

        for i in range(n):
            fy_list[i], mz_list[i] = self.compute_forces(
                fz=float(fz_flat[i]),
                alpha=float(alpha_flat[i]),
                camber=float(camber_flat[i]),
                pressure=pressure,
                velocity=velocity,
            )

        return fy_list.reshape(np.asarray(fz_arr).shape), mz_list.reshape(np.asarray(fz_arr).shape)


# Neural Network Architecture for Pacejka Head
class SimpleNN(nn.Module):
    def __init__(self, size):
        super(SimpleNN, self).__init__()
        layers = []
        for i in range(len(size) - 1):
            layers.append(nn.Linear(size[i], size[i + 1]))
            if i < len(size) - 2:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PacejkaHead(nn.Module):
    def __init__(self, model_size):
        super(PacejkaHead, self).__init__()
        assert model_size[-1] == 2
        self.FFN = SimpleNN(model_size)

    def forward(self, main_block_outs, alpha):
        params = self.FFN(main_block_outs)
        B = params[..., 0]
        D = params[..., 1]
        out = D * torch.sin(2.2 * torch.atan(torch.atan(B * alpha)))
        return out


class PacejkaNN_splitHead(nn.Module):
    def __init__(self, size, SA_index, SA_mean, SA_var, norm_params: dict, split=-2):
        super(PacejkaNN_splitHead, self).__init__()
        layers = []
        backbone_size = size[:split]
        for i in range(len(backbone_size) - 1):
            if i == 0:
                layers.append(nn.Linear(backbone_size[0] - 1, backbone_size[i + 1]))
            else:
                layers.append(nn.Linear(backbone_size[i], backbone_size[i + 1]))
            if i < len(backbone_size) - 2:
                layers.append(nn.GELU())

        self.network = nn.Sequential(*layers)
        self.SA_index = SA_index
        self.SA_mean = SA_mean
        self.SA_var = SA_var

        self.input_columns = list(range(size[0]))
        self.input_columns.remove(self.SA_index)

        head_size = size[split - 1 :]
        self.FY_head = PacejkaHead(head_size + [2])
        self.MZ_head = PacejkaHead(head_size + [2])
        self.NORM_PARAMS = norm_params

    def forward(self, x):
        denorm_SA = (x[..., self.SA_index] * self.SA_var) + self.SA_mean
        x = x[..., self.input_columns]
        params = self.network(x)
        fy = self.FY_head(params, denorm_SA)
        mz = self.MZ_head(params, denorm_SA)
        return torch.stack((fy, mz), dim=-1)


class PacejkaNNTireModel(TireModel):
    """
    Adapter and wrapper for the pretrained Pacejka Neural Network tire model.
    Converts SI standard inputs (Z-up, Fz > 0, rad) to/from internal TTC NN representations.
    """

    INPUT_COLUMNS = ["FZ", "IA", "P", "SA", "TSTC", "V", "tire_dim1", "tire_dim2"]

    def __init__(
        self,
        weights_path: Union[str, Path] = "NN_model/R20_tire_model.pth",
        norm_params_path: Union[str, Path] = "NN_model/norm_params.json",
        device: str = "cpu",
        scale_factor: float = 0.93,
    ):
        self.device = torch.device(device)
        self.scale_factor = scale_factor

        weights_p = Path(weights_path)
        norm_p = Path(norm_params_path)

        if not norm_p.exists():
            # Fallback to repo root
            root_fallback = Path(__file__).resolve().parent.parent / norm_params_path
            if root_fallback.exists():
                norm_p = root_fallback
            else:
                raise FileNotFoundError(f"Tire normalization parameters not found: {norm_p}")

        if not weights_p.exists():
            root_fallback = Path(__file__).resolve().parent.parent / weights_path
            if root_fallback.exists():
                weights_p = root_fallback
            else:
                raise FileNotFoundError(f"Tire model weights not found: {weights_p}")

        with open(norm_p, "r") as f:
            self.norm_params = json.load(f)

        sa_idx = self.INPUT_COLUMNS.index("SA")
        sa_mean = self.norm_params["SA"][0]
        sa_var = self.norm_params["SA"][1]

        self.model = PacejkaNN_splitHead(
            [len(self.INPUT_COLUMNS), 32, 32, 16],
            SA_index=sa_idx,
            SA_mean=sa_mean,
            SA_var=sa_var,
            norm_params=self.norm_params,
            split=-1,
        )

        state_dict = torch.load(weights_p, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def compute_forces(
        self,
        fz: float,
        alpha: float,
        camber: float = 0.0,
        pressure: float = 75842.0,
        velocity: float = 11.176,
    ) -> Tuple[float, float]:
        """
        Calculates Fy (N) and Mz (N*m) from positive Fz (N) and alpha (rad).
        """
        # Guard against zero or negative load (tire lifted off ground)
        if fz <= 1e-3:
            return 0.0, 0.0

        # Convert to TTC internal conventions:
        # FZ is negative in TTC format (e.g. -800 N)
        ttc_fz = -float(fz)
        # SA is degrees in TTC format
        ttc_sa = float(alpha) * (180.0 / np.pi)
        # Camber (IA) in degrees
        ttc_ia = float(camber) * (180.0 / np.pi)
        # Pressure in kPa
        ttc_p = float(pressure) / 1000.0
        # Velocity in km/h (11.176 m/s = 40.23 km/h)
        ttc_v = float(velocity) * 3.6

        # Normalization
        norm_fz = (ttc_fz - self.norm_params["FZ"][0]) / self.norm_params["FZ"][1]
        norm_sa = (ttc_sa - self.norm_params["SA"][0]) / self.norm_params["SA"][1]
        norm_ia = (ttc_ia - self.norm_params["IA"][0]) / self.norm_params["IA"][1]
        norm_p = (ttc_p - self.norm_params["P"][0]) / self.norm_params["P"][1]
        norm_v = (ttc_v - self.norm_params["V"][0]) / self.norm_params["V"][1]
        norm_tstc = 0.0  # surface temp default
        norm_d1 = 0.0
        norm_d2 = 0.0

        input_tensor = torch.zeros((1, 8), dtype=torch.float32, device=self.device)
        input_tensor[0, self.INPUT_COLUMNS.index("FZ")] = norm_fz
        input_tensor[0, self.INPUT_COLUMNS.index("SA")] = norm_sa
        input_tensor[0, self.INPUT_COLUMNS.index("IA")] = norm_ia
        input_tensor[0, self.INPUT_COLUMNS.index("P")] = norm_p
        input_tensor[0, self.INPUT_COLUMNS.index("TSTC")] = norm_tstc
        input_tensor[0, self.INPUT_COLUMNS.index("V")] = norm_v
        input_tensor[0, self.INPUT_COLUMNS.index("tire_dim1")] = norm_d1
        input_tensor[0, self.INPUT_COLUMNS.index("tire_dim2")] = norm_d2

        with torch.no_grad():
            out = self.model(input_tensor)[0]
            fy_norm = out[0].item()
            mz_norm = out[1].item()

        # Denormalize outputs
        fy = (fy_norm * self.norm_params["FY"][1]) + self.norm_params["FY"][0]
        mz = (mz_norm * self.norm_params["MZ"][1]) + self.norm_params["MZ"][0]

        # Apply scaling
        fy *= self.scale_factor
        mz *= self.scale_factor

        return float(fy), float(mz)

    def compute_forces_batch(
        self,
        fz_arr: np.ndarray,
        alpha_arr: np.ndarray,
        camber_arr: Optional[np.ndarray] = None,
        pressure: float = 75842.0,
        velocity: float = 11.176,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast vectorized batch inference for entire 2D grids."""
        fz_flat = np.asarray(fz_arr, dtype=np.float32).flatten()
        alpha_flat = np.asarray(alpha_arr, dtype=np.float32).flatten()
        n = len(fz_flat)

        if camber_arr is None:
            camber_flat = np.zeros(n, dtype=np.float32)
        else:
            camber_flat = np.asarray(camber_arr, dtype=np.float32).flatten()

        # TTC conversions
        ttc_fz = -fz_flat
        ttc_sa = alpha_flat * (180.0 / np.pi)
        ttc_ia = camber_flat * (180.0 / np.pi)
        ttc_p = pressure / 1000.0
        ttc_v = velocity * 3.6

        norm_fz = (ttc_fz - self.norm_params["FZ"][0]) / self.norm_params["FZ"][1]
        norm_sa = (ttc_sa - self.norm_params["SA"][0]) / self.norm_params["SA"][1]
        norm_ia = (ttc_ia - self.norm_params["IA"][0]) / self.norm_params["IA"][1]
        norm_p = (ttc_p - self.norm_params["P"][0]) / self.norm_params["P"][1]
        norm_v = (ttc_v - self.norm_params["V"][0]) / self.norm_params["V"][1]

        input_matrix = np.zeros((n, 8), dtype=np.float32)
        input_matrix[:, self.INPUT_COLUMNS.index("FZ")] = norm_fz
        input_matrix[:, self.INPUT_COLUMNS.index("SA")] = norm_sa
        input_matrix[:, self.INPUT_COLUMNS.index("IA")] = norm_ia
        input_matrix[:, self.INPUT_COLUMNS.index("P")] = norm_p
        input_matrix[:, self.INPUT_COLUMNS.index("V")] = norm_v

        input_tensor = torch.from_numpy(input_matrix).to(self.device)

        with torch.no_grad():
            out = self.model(input_tensor)
            fy_norm = out[:, 0].cpu().numpy()
            mz_norm = out[:, 1].cpu().numpy()

        fy = (fy_norm * self.norm_params["FY"][1]) + self.norm_params["FY"][0]
        mz = (mz_norm * self.norm_params["MZ"][1]) + self.norm_params["MZ"][0]

        fy *= self.scale_factor
        mz *= self.scale_factor

        # Zero out where normal load is non-positive
        zero_mask = fz_flat <= 1e-3
        fy[zero_mask] = 0.0
        mz[zero_mask] = 0.0

        return fy.reshape(np.asarray(fz_arr).shape), mz.reshape(np.asarray(fz_arr).shape)


def load_nn_tire_model(
    weights_path: str = "NN_model/R20_tire_model.pth",
    norm_params_path: str = "NN_model/norm_params.json",
) -> PacejkaNNTireModel:
    """Factory helper to instantiate the neural network tire model."""
    return PacejkaNNTireModel(weights_path=weights_path, norm_params_path=norm_params_path)
