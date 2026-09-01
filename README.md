# SC Racing - Yaw Moment Diagram (YMD) Solver

A modular, quasi-static vehicle dynamics simulation tool for Formula Student / Formula SAE race cars. This package computes steady-state cornering handling envelopes across a two-dimensional grid of chassis slip angles ($\beta$) and steering angles ($\delta$), evaluates key vehicle handling metrics (KPIs), and provides an interactive Streamlit dashboard for parameter exploration and sensitivity analysis.

---

## Features

- **4-Corner Quasi-Static Load Transfer**:
  - Sprung mass roll moment distribution determined by front/rear spring rates, anti-roll bars, and motion ratios.
  - Geometric unsprung weight transfer governed by independent front and rear roll center heights.
  - Aerodynamic downforce and pitch moment distribution based on dynamic speed, frontal area, $C_l$, and Center of Pressure (CoP).
- **Non-Linear Tire Model**:
  - Neural network Pacejka formulation trained on TTC tire test data (`NN_model/R20_tire_model.pth`).
  - Native SI interface converting loads ($F_z > 0\text{ N}$) and slip angles ($\text{rad}$) with vectorized batch evaluation.
- **Steady-State Equilibrium Solver**:
  - Fixed-point iteration with under-relaxation damping to solve coupled steady-state lateral acceleration ($A_y$) and yaw rate ($r = A_y / V_x$).
  - Warm-starting across adjacent grid points for rapid convergence.
- **Handling Key Performance Indicators (KPIs)**:
  - Steady-State Grip Limit ($A_y$ at $M_z = 0$).
  - Absolute Peak Lateral Acceleration ($A_{y,\max}$).
  - Limit Balance ($M_z$ at peak $A_y$).
  - Steering Control Authority ($\partial M_z / \partial \delta$ at $\beta=0, \delta=0$).
  - Yaw Stability ($\partial M_z / \partial \beta$ at $\beta=0, \delta=0$).
  - Total Lateral Load Transfer Distribution (TLLTD) and Roll Gradient.
- **Interactive Web Interface & Visualizations**:
  - Streamlit dashboard with full parameter control across mass, geometry, suspension, aero, and tires.
  - Solve-on-demand mode with session caching to prevent unnecessary re-computations during multi-parameter adjustments.
  - 1D parameter sensitivity sweeps with dedicated axes for Limit Balance.
  - Export capabilities for high-resolution PNG plots, CSV grid datasets, and custom YAML configuration files.

---

## Project Structure

```text
YMD-SCR/
├── base_params_SCR26.yaml   # Default vehicle parameters and simulation setup
├── main.py                  # CLI entrypoint for solving, sweeping, and launching GUI
├── app.py                   # Root entrypoint for Streamlit Cloud deployment
├── requirements.txt         # Python dependencies
├── NN_model/                # Pre-trained neural network tire weights and norm parameters
│   ├── R20_tire_model.pth
│   └── norm_params.json
├── src/
│   ├── app.py               # Streamlit web application
│   ├── config.py            # Dataclasses and YAML configuration loader
│   ├── kinematics.py        # 4-corner wheel kinematics and Ackermann geometry
│   ├── kpi.py               # KPI extraction and adaptive sub-grid refinement
│   ├── load_transfer.py     # 4-corner weight transfer and aerodynamic distribution
│   ├── solver.py            # Steady-state equilibrium solver and sweep routines
│   ├── tire_model.py        # Abstract tire model interface and Pacejka NN wrapper
│   └── visualization.py     # Matplotlib and Plotly charting utilities
└── tests/                   # Automated unit test suite
    ├── test_config.py
    ├── test_kinematics.py
    ├── test_load_transfer.py
    ├── test_solver.py
    └── test_tire_model.py
```

---

## Installation

Ensure Python 3.9+ is installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/YMD-SCR.git
cd YMD-SCR
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Web Dashboard
To launch the Streamlit graphical interface:

```bash
streamlit run app.py
```
Or via the CLI runner:
```bash
python3 main.py gui
```

### 2. Command-Line Solver
Solve the baseline configuration directly in the terminal and output a KPI summary table:

```bash
python3 main.py solve --config base_params_SCR26.yaml --speed 25 --save-plot outputs/ymd_25mph.png
```

### 3. Parameter Sweeps (CLI)
Run a 1D sensitivity sweep across vehicle speed:

```bash
python3 main.py sweep --param velocity --min 15 --max 60 --points 10 --save-plot outputs/sweep_speed.png
```

---

## Running Tests

Run the test suite to verify solver mechanics, kinematics, and load transfer calculations:

```bash
python3 -m unittest discover -s tests
```

---

## Coordinate System and Sign Conventions

- **Vehicle Coordinate System**: ISO 8855 / SAE standard (X forward, Y left, Z up).
- **Chassis Slip Angle ($\beta$)**: Angle between the vehicle centerline and velocity vector (counter-clockwise positive).
- **Steer Angle ($\delta$)**: Road wheel steering angle (counter-clockwise / left positive).
- **Yaw Moment ($M_z$)**: Counter-clockwise positive. A negative $M_z$ during a positive lateral acceleration turn indicates understeer; a positive $M_z$ indicates oversteer.
- **Normal Load ($F_z$)**: Strictly positive in compression ($F_z > 0\text{ N}$).
