"""Figures, rendered as PNG previews.

No panel draws event lines or shaded periods: there is no code path
here that can add a vertical rule, an annotation or a shaded span.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mardynamics.application.emp_config as emp_config

# A colour-blind-safe cycle used for every panel.
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)


def decimal_year(period: str) -> float:
    """Convert ``YYYY-MM`` to a decimal year for a plain calendar axis.

    Parameters
    ----------
    period : str
        The month.

    Returns
    -------
    float
        Year plus the zero-based month fraction.
    """
    year, month = str(period).split("-")
    return int(year) + (int(month) - 1) / 12.0


def _render_png(panels: list[dict], destination, columns: int = 2) -> None:
    """Render the same series to a PNG preview.

    Parameters
    ----------
    panels : list of dict
        Each with ``title``, ``xlabel``, ``ylabel``, ``series`` and optionally
        ``log``.
    destination : pathlib.Path
        Output path.
    columns : int, default 2
        Panels per row.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(6.0 * columns, 3.4 * rows), squeeze=False
    )
    for index, panel in enumerate(panels):
        axis = axes[index // columns][index % columns]
        for order, (label, x, y) in enumerate(panel["series"]):
            axis.plot(
                np.asarray(x, dtype=float),
                np.asarray(y, dtype=float),
                label=label,
                color=PALETTE[order % len(PALETTE)],
                linewidth=1.2,
            )
        axis.set_title(panel["title"], fontsize=10)
        axis.set_xlabel(panel["xlabel"], fontsize=9)
        axis.set_ylabel(panel["ylabel"], fontsize=9)
        if panel.get("log"):
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        if len(panel["series"]) > 1:
            axis.legend(fontsize=7, frameon=False)
    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_figure(paths, name: str, panels: list[dict], columns: int = 2):
    """Write one figure as a PNG preview.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    name : str
        Figure stem.
    panels : list of dict
        The panels.
    columns : int, default 2
        Panels per row.

    Returns
    -------
    pathlib.Path
        The PNG preview.
    """
    png = paths.figures / f"{name}.png"
    _render_png(panels, png, columns)
    return png


# =========================================================
# The seven figures
# =========================================================
def influence_paths(paths, frame: pd.DataFrame):
    """Figure ``fig_influence_paths``: the four coordinate-invariant paths.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        ``influence_paths.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    panels = []
    for name, title in (
        ("global", "Global"),
        ("row", "Row shape"),
        ("column", "Column shape"),
        ("amplitude", "Amplitude"),
    ):
        block = frame[frame["path"] == name].sort_values("observation_index")
        panels.append(
            {
                "title": title,
                "xlabel": "Year",
                "ylabel": "Standardised influence",
                "series": [
                    (
                        title,
                        [decimal_year(p) for p in block["period"]],
                        block["value"].to_numpy(dtype=float),
                    )
                ],
            }
        )
    return write_figure(paths, "fig_influence_paths", panels)


def validation(paths, tuning: pd.DataFrame):
    """Figure ``fig_validation``: the four tuning panels.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    tuning : pandas.DataFrame
        ``validation_tuning.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    feasible = tuning[tuning["feasible"]]

    static = feasible[feasible["model"] == "static_reduced"].sort_values("rank")
    rolling = feasible[feasible["model"] == "rolling_full"].sort_values("window")
    ms = feasible[feasible["model"] == "ms_reduced"].sort_values("rank")
    surface = feasible[feasible["model"] == "rolling_reduced"]

    surface_series = []
    for rank in emp_config.RANK_GRID:
        block = surface[surface["rank"] == rank].sort_values("window")
        if not block.empty:
            surface_series.append(
                (
                    f"$r={rank}$",
                    block["window"].to_numpy(dtype=float),
                    block["validation_msfe"].to_numpy(dtype=float),
                )
            )

    panels = [
        {
            "title": "Static reduced rank",
            "xlabel": "Rank $r$",
            "ylabel": "Validation MSFE",
            "series": [
                (
                    "Static reduced rank",
                    static["rank"].to_numpy(dtype=float),
                    static["validation_msfe"].to_numpy(dtype=float),
                )
            ],
        },
        {
            "title": "Rolling full rank",
            "xlabel": "Window $w$ (months)",
            "ylabel": "Validation MSFE",
            "series": [
                (
                    "Rolling full rank",
                    rolling["window"].to_numpy(dtype=float),
                    rolling["validation_msfe"].to_numpy(dtype=float),
                )
            ],
        },
        {
            "title": "Rolling reduced rank $(w,r)$",
            "xlabel": "Window $w$ (months)",
            "ylabel": "Validation MSFE",
            "series": surface_series,
        },
        {
            "title": "MS-MAR reduced rank",
            "xlabel": "Rank $r$",
            "ylabel": "Validation MSFE",
            "series": [
                (
                    "MS-MAR reduced rank",
                    ms["rank"].to_numpy(dtype=float),
                    ms["validation_msfe"].to_numpy(dtype=float),
                )
            ],
        },
    ]
    return write_figure(paths, "fig_validation", panels)


def period_loss(paths, frame: pd.DataFrame):
    """Figure ``fig_period_loss``: the logarithmic monthly loss path.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        ``period_loss.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    series = []
    for model in emp_config.MODELS:
        block = frame[frame["model"] == model].sort_values("period")
        series.append(
            (
                emp_config.MODEL_LABELS[model],
                [decimal_year(p) for p in block["period"]],
                block["loss"].to_numpy(dtype=float),
            )
        )
    panels = [
        {
            "title": "Monthly forecast loss",
            "xlabel": "Year",
            "ylabel": "Squared Frobenius error",
            "series": series,
            "log": True,
        }
    ]
    return write_figure(paths, "fig_period_loss", panels, columns=1)


def rolling_path(paths, frame: pd.DataFrame):
    """Figure ``fig_rolling_path``: the forecast-selected rolling path.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        ``rolling_canonical_path.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    ordered = frame.sort_values("period")
    years = [decimal_year(p) for p in ordered["period"]]
    panels = [
        {
            "title": "Shape distance to the test-scale benchmark",
            "xlabel": "Year",
            "ylabel": "Frobenius distance",
            "series": [
                (
                    "Row shape",
                    years,
                    ordered[
                        "row_shape_distance_to_test_scale_benchmark"
                    ].to_numpy(dtype=float),
                ),
                (
                    "Column shape",
                    years,
                    ordered[
                        "column_shape_distance_to_test_scale_benchmark"
                    ].to_numpy(dtype=float),
                ),
            ],
        },
        {
            "title": "Amplitude and persistence",
            "xlabel": "Year",
            "ylabel": "Value",
            "series": [
                ("Amplitude $\\gamma_t$", years, ordered["gamma"].to_numpy(dtype=float)),
                (
                    "Joint spectral radius",
                    years,
                    ordered["joint_spectral_radius"].to_numpy(dtype=float),
                ),
            ],
        },
    ]
    return write_figure(paths, "fig_rolling_path", panels)


def rolling_increments(paths, frame: pd.DataFrame):
    """Figure ``fig_rolling_increments``: the canonical increments.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        ``rolling_increments.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    ordered = frame.sort_values("period")
    years = [decimal_year(p) for p in ordered["period"]]
    panels = [
        {
            "title": "Shape increments",
            "xlabel": "Year",
            "ylabel": "Frobenius increment",
            "series": [
                ("$d_{\\mathrm{row}}(t)$", years, ordered["d_row"].to_numpy(dtype=float)),
                (
                    "$d_{\\mathrm{column}}(t)$",
                    years,
                    ordered["d_column"].to_numpy(dtype=float),
                ),
            ],
        },
        {
            "title": "Amplitude increment",
            "xlabel": "Year",
            "ylabel": "$|\\Delta\\log\\gamma_t|$",
            "series": [
                ("$d_\\gamma(t)$", years, ordered["d_gamma"].to_numpy(dtype=float))
            ],
        },
    ]
    return write_figure(paths, "fig_rolling_increments", panels)


def regime_smoothed(paths, frame: pd.DataFrame):
    """Figure ``fig_regime_smoothed``: full-sample smoothed probabilities.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        ``regime_probabilities_smoothed.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    high = frame[frame["regime_label"] == "high_variance"].sort_values("period")
    panels = [
        {
            "title": "Smoothed probability of the high-variance regime",
            "xlabel": "Year",
            "ylabel": "Probability",
            "series": [
                (
                    "Smoothed",
                    [decimal_year(p) for p in high["period"]],
                    high["smoothed_probability"].to_numpy(dtype=float),
                )
            ],
        }
    ]
    return write_figure(paths, "fig_regime_smoothed", panels, columns=1)


def regime_oos(paths, frame: pd.DataFrame):
    """Figure ``fig_regime_oos``: ex-ante predicted probabilities.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        ``regime_probabilities_oos.csv``.

    Returns
    -------
    pathlib.Path
        The written output.
    """
    high = frame[frame["regime_label"] == "high_variance"].sort_values("period")
    panels = [
        {
            "title": "Ex-ante predicted probability of the high-variance regime",
            "xlabel": "Year",
            "ylabel": "Probability",
            "series": [
                (
                    "One-step predicted",
                    [decimal_year(p) for p in high["period"]],
                    high["predicted_probability"].to_numpy(dtype=float),
                )
            ],
        }
    ]
    return write_figure(paths, "fig_regime_oos", panels, columns=1)
