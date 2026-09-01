"""
Unit tests for vehicle configuration parsing and unit conversion.
"""

import unittest
from pathlib import Path
from src.config import (
    load_config,
    INCH_TO_M,
    LB_TO_KG,
    LBF_PER_IN_TO_N_PER_M,
    MPH_TO_MS,
    G_ACCEL,
)


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config_path = "base_params_SCR26.yaml"

    def test_load_config_values(self):
        cfg = load_config(self.config_path)

        # 1. Total Mass: 490 dry + 150 driver + 10 fuel = 650 lb -> ~294.835 kg
        expected_mass_kg = 650.0 * LB_TO_KG
        self.assertAlmostEqual(cfg.mass.total_mass, expected_mass_kg, places=2)

        # 2. Wheelbase: 60.5 in -> 1.5367 m
        expected_wb_m = 60.5 * INCH_TO_M
        self.assertAlmostEqual(cfg.dimensions.wheelbase, expected_wb_m, places=3)

        # 3. CG to front axle (a) and rear axle (b): x_loc = 0.49 front -> a = wb * 0.51, b = wb * 0.49
        self.assertAlmostEqual(cfg.a, expected_wb_m * 0.51, places=3)
        self.assertAlmostEqual(cfg.b, expected_wb_m * 0.49, places=3)

        # 4. Track width: 48 in -> 1.2192 m
        expected_tw_m = 48.0 * INCH_TO_M
        self.assertAlmostEqual(cfg.front_suspension.track_width, expected_tw_m, places=3)
        self.assertAlmostEqual(cfg.rear_suspension.track_width, expected_tw_m, places=3)

        # 5. Spring rates: Front 300 lb/in, Rear 250 lb/in
        expected_f_spring = 300.0 * LBF_PER_IN_TO_N_PER_M
        expected_r_spring = 250.0 * LBF_PER_IN_TO_N_PER_M
        self.assertAlmostEqual(cfg.front_suspension.spring_rate, expected_f_spring, places=1)
        self.assertAlmostEqual(cfg.rear_suspension.spring_rate, expected_r_spring, places=1)

        # 6. Simulation speed: 25 mph -> 11.176 m/s
        expected_speed_ms = 25.0 * MPH_TO_MS
        self.assertAlmostEqual(cfg.simulation.velocity, expected_speed_ms, places=2)


if __name__ == "__main__":
    unittest.main()
