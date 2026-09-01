"""
Visualization Engine for Yaw Moment Diagrams (YMD)
Provides publication-ready high-resolution Matplotlib plots and interactive Plotly figures.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from .solver import YMDResult
from .kpi import VehicleKPIs, KPICalculator


class YMDVisualizer:
    """
    Renders high-quality Yaw Moment Diagrams (YMD) with isolines, KPI annotations,
    and interactive inspection tools.
    """

    @staticmethod
    def plot_matplotlib(
        result: YMDResult,
        kpis: Optional[VehicleKPIs] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        figsize: tuple = (11, 7.5),
    ) -> plt.Figure:
        """
        Generates a publication-grade Matplotlib Yaw Moment Diagram.
        """
        if kpis is None:
            kpis = KPICalculator(result).compute_all_kpis()

        ay_g = result.ay_grid_g
        mz = result.mz_grid
        beta_deg = result.beta_grid_deg[:, 0]
        delta_deg = result.delta_grid_deg[0, :]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Background grid & axes
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        # Colormaps for isolines
        cmap_delta = cm.get_cmap("coolwarm", len(delta_deg))
        cmap_beta = cm.get_cmap("viridis", len(beta_deg))

        # 1. Plot lines of constant Beta (varying Delta)
        for i, beta in enumerate(beta_deg):
            is_zero = np.isclose(beta, 0.0, atol=1e-3)
            color = "#990000" if is_zero else cmap_beta(i)
            lw = 2.5 if is_zero else 1.2
            alpha = 1.0 if is_zero else 0.65
            label = "β = 0°" if is_zero else None
            ax.plot(
                ay_g[i, :],
                mz[i, :],
                color=color,
                linewidth=lw,
                alpha=alpha,
                linestyle="-",
                label=label,
            )

        # 2. Plot lines of constant Delta (varying Beta)
        for j, delta in enumerate(delta_deg):
            is_zero = np.isclose(delta, 0.0, atol=1e-3)
            color = "#D4A017" if is_zero else cmap_delta(j)
            lw = 2.5 if is_zero else 1.0
            alpha = 1.0 if is_zero else 0.55
            label = "δ = 0°" if is_zero else None
            ax.plot(
                ay_g[:, j],
                mz[:, j],
                color=color,
                linewidth=lw,
                alpha=alpha,
                linestyle="--",
                label=label,
            )

        # 3. Annotate KPIs
        # Steady state grip limit
        ax.scatter(
            [kpis.steady_state_grip_limit_g],
            [0.0],
            color="#990000",
            s=90,
            zorder=6,
            edgecolors="#FFCC00",
            linewidths=1.5,
            label=f"Grip Limit ({kpis.steady_state_grip_limit_g:.2f}g)",
        )

        # Limit balance point (max Ay)
        ax.scatter(
            [kpis.max_ay_g],
            [kpis.limit_balance_nm],
            color="#FFCC00",
            s=90,
            zorder=6,
            edgecolors="#990000",
            linewidths=1.5,
            label=f"Max Ay ({kpis.max_ay_g:.2f}g, Mz={kpis.limit_balance_nm:.0f} N*m)",
        )

        # Formatting
        chart_title = (
            title
            if title is not None
            else f"Yaw Moment Diagram @ {result.velocity_mph:.1f} mph ({result.velocity_ms:.1f} m/s)"
        )
        ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=12, color="#990000")
        ax.set_xlabel("Lateral Acceleration Ay [g]", fontsize=12, fontweight="medium")
        ax.set_ylabel("Yaw Moment Mz [N*m]", fontsize=12, fontweight="medium")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Add KPI Summary Box
        kpi_text = (
            f"--- Performance KPIs ---\n"
            f"Grip Limit (Mz=0): {kpis.steady_state_grip_limit_g:.2f} g\n"
            f"Max Lateral Accel: {kpis.max_ay_g:.2f} g\n"
            f"Limit Balance: {kpis.limit_balance_nm:+.1f} N*m ({kpis.limit_balance_type})\n"
            f"Control Authority: {kpis.control_nm_per_deg:+.1f} N*m/deg\n"
            f"Yaw Stability: {kpis.stability_nm_per_deg:+.1f} N*m/deg\n"
            f"Front TLLTD: {kpis.tlltd_front_percent:.1f} %\n"
            f"Roll Gradient: {kpis.roll_gradient_deg_per_g:.2f} deg/g"
        )
        props = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#990000", alpha=0.9)
        ax.text(
            0.02,
            0.97,
            kpi_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=props,
            zorder=7,
        )

        ax.legend(loc="lower left", framealpha=0.9, fontsize=9.5)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=dpi, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_plotly(
        result: YMDResult,
        kpis: Optional[VehicleKPIs] = None,
        title: Optional[str] = None,
        compare_result: Optional[YMDResult] = None,
        compare_label: str = "Modified Setup",
        base_label: str = "Baseline",
    ) -> go.Figure:
        """
        Generates an interactive Plotly Yaw Moment Diagram with rich hover tooltips.
        """
        if kpis is None:
            kpis = KPICalculator(result).compute_all_kpis()

        ay_g = result.ay_grid_g
        mz = result.mz_grid
        beta_deg = result.beta_grid_deg[:, 0]
        delta_deg = result.delta_grid_deg[0, :]

        fig = go.Figure()

        # 1. Constant Beta curves (varying Delta)
        for i, beta in enumerate(beta_deg):
            is_zero = np.isclose(beta, 0.0, atol=1e-3)
            hover_text = [
                f"Beta (Chassis Slip): {beta:.1f}°<br>"
                f"Delta (Steer): {delta_deg[j]:.1f}°<br>"
                f"Ay: {ay_g[i, j]:.3f} g ({ay_g[i, j]*9.81:.2f} m/s²)<br>"
                f"Mz: {mz[i, j]:.1f} N*m<br>"
                f"FL Load: {result.point_solutions[i][j].corner_loads.FL:.0f} N<br>"
                f"FR Load: {result.point_solutions[i][j].corner_loads.FR:.0f} N"
                for j in range(len(delta_deg))
            ]

            line_color = "#990000" if is_zero else "rgba(92, 92, 92, 0.35)"
            line_width = 3.5 if is_zero else 1.2

            fig.add_trace(
                go.Scatter(
                    x=ay_g[i, :],
                    y=mz[i, :],
                    mode="lines",
                    name=f"β = {beta:.1f}°" if (is_zero or i % 3 == 0) else "",
                    line=dict(color=line_color, width=line_width),
                    hoverinfo="text",
                    text=hover_text,
                    showlegend=bool(is_zero or (i % 3 == 0)),
                )
            )

        # 2. Constant Delta curves (varying Beta)
        for j, delta in enumerate(delta_deg):
            is_zero = np.isclose(delta, 0.0, atol=1e-3)
            hover_text = [
                f"Delta (Steer): {delta:.1f}°<br>"
                f"Beta (Chassis Slip): {beta_deg[i]:.1f}°<br>"
                f"Ay: {ay_g[i, j]:.3f} g<br>"
                f"Mz: {mz[i, j]:.1f} N*m"
                for i in range(len(beta_deg))
            ]

            line_color = "#D4A017" if is_zero else "rgba(153, 0, 0, 0.30)"
            line_width = 3.5 if is_zero else 1.2

            fig.add_trace(
                go.Scatter(
                    x=ay_g[:, j],
                    y=mz[:, j],
                    mode="lines",
                    name=f"δ = {delta:.1f}°" if (is_zero or j % 3 == 0) else "",
                    line=dict(color=line_color, width=line_width, dash="dash" if not is_zero else "solid"),
                    hoverinfo="text",
                    text=hover_text,
                    showlegend=bool(is_zero or (j % 3 == 0)),
                )
            )

        # 3. Optional Comparison Overlay
        if compare_result is not None:
            comp_ay = compare_result.ay_grid_g
            comp_mz = compare_result.mz_grid
            fig.add_trace(
                go.Scatter(
                    x=comp_ay.flatten(),
                    y=comp_mz.flatten(),
                    mode="markers",
                    name=compare_label,
                    marker=dict(size=4, color="#5C5C5C", opacity=0.4),
                )
            )

        # KPI Markers
        fig.add_trace(
            go.Scatter(
                x=[kpis.steady_state_grip_limit_g],
                y=[0.0],
                mode="markers+text",
                name=f"Grip Limit ({kpis.steady_state_grip_limit_g:.2f}g)",
                text=["Grip Limit"],
                textposition="top right",
                marker=dict(size=12, color="#990000", symbol="diamond", line=dict(width=2, color="#FFCC00")),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[kpis.max_ay_g],
                y=[kpis.limit_balance_nm],
                mode="markers+text",
                name=f"Max Ay ({kpis.max_ay_g:.2f}g)",
                text=["Max Ay"],
                textposition="bottom right",
                marker=dict(size=12, color="#FFCC00", symbol="circle", line=dict(width=2, color="#990000")),
            )
        )

        # Zero lines
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dot", line_color="gray")

        chart_title = (
            title
            if title is not None
            else f"Yaw Moment Diagram @ {result.velocity_mph:.1f} mph"
        )

        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=18)),
            xaxis_title="Lateral Acceleration Ay [g]",
            yaxis_title="Yaw Moment Mz [N*m]",
            template="plotly_white",
            hovermode="closest",
            width=950,
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        )

        return fig
