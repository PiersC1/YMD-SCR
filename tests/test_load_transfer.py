"""
Unit tests for 4-corner Load Transfer and Aero Downforce.
"""

import unittest
from src.config import load_config, G_ACCEL
from src.load_transfer import LoadTransferModel


class TestLoadTransfer(unittest.TestCase):

    def setUp(self):
        self.config = load_config("base_params_SCR26.yaml")
        self.lt = LoadTransferModel(self.config)

    def test_static_weight_conservation(self):
        # Sum of 4 corner static loads should equal total vehicle weight m * g
        total_weight = self.config.mass.total_mass * G_ACCEL
        sum_corners = (
            self.lt.fz_static_fl
            + self.lt.fz_static_fr
            + self.lt.fz_static_rl
            + self.lt.fz_static_rr
        )
        self.assertAlmostEqual(sum_corners, total_weight, places=2)

    def test_aero_downforce_at_speed(self):
        # At 11.176 m/s (25 mph)
        fz_f, fz_r, q = self.lt.compute_aero_loads(11.176)
        self.assertGreater(q, 100.0)  # Should produce meaningful downforce
        self.assertAlmostEqual(fz_f + fz_r, q, places=2)

    def test_lateral_weight_transfer_direction(self):
        # Under positive Ay (left turn), right side tires (FR, RR) gain load, left side loses load
        loads, axle_lt = self.lt.compute_load_transfer(ay=10.0, velocity=11.176)
        self.assertGreater(loads.FR, loads.FL)
        self.assertGreater(loads.RR, loads.RL)
        self.assertGreater(axle_lt.roll_angle, 0.0)
        self.assertGreater(axle_lt.tlltd, 0.0)
        self.assertLess(axle_lt.tlltd, 100.0)


if __name__ == "__main__":
    unittest.main()
