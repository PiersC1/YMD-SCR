"""
Unit tests for Kinematics & Ackermann Steering.
"""

import unittest
import numpy as np
from src.config import load_config
from src.kinematics import KinematicsModel


class TestKinematics(unittest.TestCase):

    def setUp(self):
        self.config = load_config("base_params_SCR26.yaml")
        self.kin = KinematicsModel(self.config)

    def test_zero_steer(self):
        delta_fl, delta_fr = self.kin.compute_steer_angles(0.0)
        self.assertEqual(delta_fl, 0.0)
        self.assertEqual(delta_fr, 0.0)

    def test_ackermann_steer_direction(self):
        # Left turn (positive delta): inner wheel (FL) turns more than outer wheel (FR) under >100% Ackermann
        delta_test = np.deg2rad(15.0)
        delta_fl, delta_fr = self.kin.compute_steer_angles(delta_test)
        self.assertGreater(delta_fl, 0.0)
        self.assertGreater(delta_fr, 0.0)
        self.assertGreater(delta_fl, delta_fr)  # Inner wheel angle > Outer wheel angle

    def test_straight_line_slip_angles(self):
        # Straight line driving at 11 m/s (beta=0, yaw_rate=0, delta=0)
        state = self.kin.compute_slip_angles(velocity=11.0, beta=0.0, yaw_rate=0.0, delta=0.0)
        # Slip angles should only reflect static toe
        self.assertAlmostEqual(state.vx_cg, 11.0, places=2)
        self.assertAlmostEqual(state.vy_cg, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
