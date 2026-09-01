"""
Unit tests for the SI-standardized Tire Model wrapper and neural network inference.
"""

import unittest
import numpy as np
from src.tire_model import load_nn_tire_model


class TestTireModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tire = load_nn_tire_model()

    def test_zero_load(self):
        """Zero vertical load should return 0 lateral force and moment."""
        fy, mz = self.tire.compute_forces(fz=0.0, alpha=np.deg2rad(5.0))
        self.assertEqual(fy, 0.0)
        self.assertEqual(mz, 0.0)

    def test_zero_slip_angle(self):
        """At zero slip angle and zero camber, lateral force should be near zero."""
        fy, mz = self.tire.compute_forces(fz=700.0, alpha=0.0, camber=0.0)
        self.assertAlmostEqual(fy, 0.0, delta=25.0)

    def test_force_sign_and_magnitude(self):
        """Positive normal load with slip angle should generate substantial grip."""
        # 800 N normal load at 6 degrees slip angle (~0.105 rad)
        fy, mz = self.tire.compute_forces(fz=800.0, alpha=np.deg2rad(6.0))
        # At 800 N vertical load, lateral grip for R20 FSAE tire is typically 800-1400 N (friction coeff ~ 1.0 to 1.7)
        self.assertGreater(abs(fy), 400.0)
        self.assertLess(abs(fy), 2500.0)

    def test_batch_inference_consistency(self):
        """Vectorized batch computation should match sequential single-point evaluations."""
        fz_arr = np.array([500.0, 750.0, 1000.0])
        sa_arr = np.deg2rad(np.array([-4.0, 0.0, 6.0]))

        fy_batch, mz_batch = self.tire.compute_forces_batch(fz_arr, sa_arr)

        for i in range(len(fz_arr)):
            fy_single, mz_single = self.tire.compute_forces(fz=fz_arr[i], alpha=sa_arr[i])
            self.assertAlmostEqual(fy_batch[i], fy_single, places=3)
            self.assertAlmostEqual(mz_batch[i], mz_single, places=3)


if __name__ == "__main__":
    unittest.main()
