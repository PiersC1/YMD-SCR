"""
Unit tests for full YMD Solver and KPI calculation.
"""

import unittest
import numpy as np
from src.config import load_config
from src.tire_model import load_nn_tire_model
from src.solver import YMDSolver
from src.kpi import KPICalculator


class TestSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("base_params_SCR26.yaml")
        cls.tire_model = load_nn_tire_model()
        cls.solver = YMDSolver(cls.config, tire_model=cls.tire_model)

    def test_single_point_origin(self):
        """At zero slip and zero steer, lateral acceleration and yaw moment should be near zero."""
        sol = self.solver.solve_point(beta=0.0, delta=0.0)
        self.assertTrue(sol.converged)
        self.assertAlmostEqual(sol.ay_g, 0.0, delta=0.05)
        self.assertAlmostEqual(sol.mz, 0.0, delta=50.0)

    def test_grid_solve_and_kpis(self):
        """Solve a complete grid and verify realistic FSAE vehicle dynamics KPIs."""
        res = self.solver.solve_grid()
        self.assertEqual(res.ay_grid_g.shape, res.mz_grid.shape)
        self.assertTrue(np.all(res.convergence_mask))

        kpis = KPICalculator(res).compute_all_kpis()

        # Steady state grip limit for SCR FSAE car at 25 mph is typically 1.5 - 2.5 g
        self.assertGreater(kpis.steady_state_grip_limit_g, 1.2)
        self.assertLess(kpis.steady_state_grip_limit_g, 3.0)

        # Control and stability gradients should be non-zero and finite
        self.assertNotEqual(kpis.control_nm_per_deg, 0.0)
        self.assertNotEqual(kpis.stability_nm_per_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
