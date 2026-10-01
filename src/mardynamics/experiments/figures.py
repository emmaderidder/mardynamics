"""The influence-path signatures.

One family of figures: the influence-path signatures drawn for one
row-shape abrupt break and one row-shape gradual drift at the medium
local departure. Each figure-data file carries the theoretical
population mean signature, the Monte Carlo mean path, the pointwise
interval of the individual simulated paths, and one pre-specified null
realisation: one row per fraction ``r``, which is what a figure is
drawn from.

The agreement measures of the same two cells (the regression
coefficient of the Monte Carlo mean on the theoretical signature, the
correlation, the root-mean-squared difference and the longer-sample
attenuation) are per-cell scalars, not per-fraction series, so they are
written to the separate ``influence_path_signature_agreement`` table by
:func:`signature_agreement` instead of repeated down every row of the
figure data.

The one retained pointwise interval is:

``path_low`` / ``path_high``
    the 2.5th and 97.5th percentiles of the individual simulated paths,
    that is, where a single replication lands.

There is no pointwise interval around the Monte Carlo mean itself. The
Monte Carlo mean, the theoretical population signature and the null
realisation are the three curves drawn.

The null realisation carries no classification label and no break date,
and none is inferred from any of these figures. The realised path of a
Brownian bridge can look structured under the null, which is why the
null realisation is drawn beside the alternative, not as a contrast to
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mardynamics.experiments.design as design
import mardynamics.experiments.dgp as dgp
import mardynamics.experiments.estimators as estimators
import mardynamics.experiments.innovations as innovations
import mardynamics.experiments.sim_config as simcfg
import mardynamics.experiments.sim_io as simio
import mardynamics.core.structural_tests as structural_tests
from mardynamics.experiments.designs import common, detection

# What the retained pointwise interval column means, carried in the
# figure data so the reading is stated on the artefact itself.
INTERVAL_DEFINITIONS = {
    "path_low_high": "distribution of individual simulated paths (2.5th-97.5th percentile)",
}


def _stacked_paths(records: pd.DataFrame) -> np.ndarray:
    """Stack the stored projected influence paths of a cell.

    Parameters
    ----------
    records : pandas.DataFrame
        The cell's stored records.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_successful, N)``.
    """
    successful = records[records["status"] == "ok"] if "status" in records else records
    paths = [
        np.asarray(value, dtype=float) for value in successful["influence_projection"]
    ]
    return np.vstack(paths) if paths else np.empty((0, 0))


def null_realisation(cell, context: dict, null_cell) -> np.ndarray:
    """Project one pre-specified null dataset on the same planted direction.

    The dataset is the pre-chosen replication of the reference isotropic null
    cell, fixed before any result was seen. It is simulated from its own stored
    seed, so nothing about it depends on this figure being drawn.

    Parameters
    ----------
    cell : Cell
        The detection cell whose direction the projection uses.
    context : dict
        That cell's context.
    null_cell : Cell
        The reference null cell supplying the dataset.

    Returns
    -------
    numpy.ndarray
        The projected standardised path of the null realisation.
    """
    truth = design.build_truth(
        null_cell.m, null_cell.n, null_cell.persistence, null_cell.configuration
    )
    tensor = dgp.simulate_static_tensor(
        truth,
        null_cell.T,
        innovations.build_innovation(null_cell.innovation, simcfg.SIGMA),
        common.dataset_rng(null_cell, simcfg.INFLUENCE_FIGURE_NULL_REPLICATION),
    )
    A, B = estimators.static_fit(tensor)
    quantities = structural_tests.chart_score_quantities(tensor, A, B)
    return detection.projected_influence_path(
        quantities, context["direction"], cell.block
    )


def influence_path_data(cell, records: pd.DataFrame, context: dict, null_path):
    """Build the figure data of one influence-path cell.

    Parameters
    ----------
    cell : Cell
        The detection cell being drawn.
    records : pandas.DataFrame
        Its stored records.
    context : dict
        Its context, for the planted direction and the designed departure.
    null_path : numpy.ndarray or None
        The pre-specified null realisation, or None to omit it.

    Returns
    -------
    pandas.DataFrame
        The tidy figure data, or an empty frame when nothing is stored.

    Notes
    -----
    The written column names are read by name when the thesis figure is
    drawn. They are, in the order a plot needs them:

    ===========================  ==================================================
    column                       series
    ===========================  ==================================================
    ``r``                        sample fraction, the x axis
    ``path_low`` / ``path_high`` 2.5th/97.5th percentile of individual paths
    ``monte_carlo_mean``         the Monte Carlo mean path
    ``theoretical_population_mean``  the theoretical population signature
    ``null_realisation``         one pre-specified realisation under stability
    ===========================  ==================================================

    These names are fixed here: the file is a required output and a
    completed production run already carries them, so they may not drift.
    """
    paths = _stacked_paths(records)
    if paths.size == 0:
        return pd.DataFrame()

    n_replications, N = paths.shape
    r = np.arange(1, N + 1) / N

    mean_path = paths.mean(axis=0)
    path_low = np.quantile(paths, 0.025, axis=0)
    path_high = np.quantile(paths, 0.975, axis=0)

    theoretical = detection.theoretical_projection(
        cell, context, detection.population_block_covariance(context, cell), r
    )

    return pd.DataFrame(
        {
            "cell_id": cell.cell_id,
            "changing_block": cell.block,
            "path": cell.path,
            "magnitude": cell.magnitude,
            "r": r,
            "theoretical_population_mean": theoretical,
            "monte_carlo_mean": mean_path,
            "path_low": path_low,
            "path_high": path_high,
            "null_realisation": (
                np.nan if null_path is None else np.asarray(null_path, dtype=float)
            ),
            "n_replications": n_replications,
            "path_interval_meaning": INTERVAL_DEFINITIONS["path_low_high"],
        }
    )


def draw_influence_path(frame: pd.DataFrame, cell):
    """Draw the combined influence-path figure of one cell.

    Parameters
    ----------
    frame : pandas.DataFrame
        The figure data.
    cell : Cell
        The cell being drawn.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    r = frame["r"].to_numpy()

    axis.fill_between(
        r,
        frame["path_low"],
        frame["path_high"],
        alpha=0.20,
        label="2.5th-97.5th percentile of individual simulated paths",
    )
    axis.plot(r, frame["monte_carlo_mean"], linewidth=2.0, label="Monte Carlo mean")
    axis.plot(
        r,
        frame["theoretical_population_mean"],
        linestyle="--",
        linewidth=2.0,
        label="theoretical population mean signature",
    )
    if frame["null_realisation"].notna().any():
        axis.plot(
            r,
            frame["null_realisation"],
            linewidth=1.0,
            alpha=0.8,
            label="one pre-specified null realisation",
        )
    axis.axhline(0.0, color="0.6", linewidth=0.8)

    axis.set_xlabel("sample fraction r")
    axis.set_ylabel("projected standardised influence path")
    axis.set_title(
        f"{cell.block}-shape {cell.path}, local {cell.magnitude} departure "
        f"({cell.m}x{cell.n}, T={cell.T}, "
        f"{int(frame['n_replications'].iloc[0])} replications)",
        fontsize=10,
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def signature_agreement(frame: pd.DataFrame) -> dict:
    """Agreement of the Monte Carlo mean with the theoretical signature.

    Reports the regression coefficient of the mean on the theoretical
    signature, the correlation and the root-mean-squared difference, and
    flags a sign or scale error against the configured thresholds.

    Parameters
    ----------
    frame : pandas.DataFrame
        The figure data of one cell.

    Returns
    -------
    dict
        The agreement measures and the flag.
    """
    theoretical = frame["theoretical_population_mean"].to_numpy()
    simulated = frame["monte_carlo_mean"].to_numpy()

    correlation = float(np.corrcoef(theoretical, simulated)[0, 1])
    slope = float(theoretical @ simulated / (theoretical @ theoretical))
    rmsd = float(np.sqrt(np.mean((simulated - theoretical) ** 2)))

    low, high = simcfg.SIGNATURE_SLOPE_BAND
    return {
        "cell_id": frame["cell_id"].iloc[0],
        "changing_block": frame["changing_block"].iloc[0],
        "path": frame["path"].iloc[0],
        "magnitude": frame["magnitude"].iloc[0],
        "n_replications": int(frame["n_replications"].iloc[0]),
        "regression_coefficient": slope,
        "correlation": correlation,
        "root_mean_squared_difference": rmsd,
        "min_correlation": simcfg.SIGNATURE_MIN_CORRELATION,
        "slope_band_low": low,
        "slope_band_high": high,
        "flag_signature": bool(
            correlation < simcfg.SIGNATURE_MIN_CORRELATION or not (low <= slope <= high)
        ),
    }


def attenuation_at_longer_sample(cell, n_rep: int | None = None) -> dict:
    """Measure the same signature slope at a longer sample length.

    The slope of the Monte Carlo mean on the theoretical signature sits
    below one at the reference sample length. Two things pull it down
    and both shrink with the sample: the path is standardised by an
    estimated block covariance instead of the population one, and the
    departure, while local in form, is not infinitesimal at a finite
    sample. Reporting the slope at a longer sample beside the reference
    one turns a flag into a measurement.

    Nothing about the design changes: the same population, the same planted
    direction, the same path and the same departure size, at one different
    sample length.

    Parameters
    ----------
    cell : Cell
        The influence-path cell whose attenuation is measured.
    n_rep : int or None, optional
        Replications; the configured diagnostic count when omitted.

    Returns
    -------
    dict
        The longer-sample slope, correlation and root-mean-squared difference,
        with the sample length and replication count they were measured at.
    """
    import mardynamics.experiments.registry as registry

    n_rep = simcfg.SIGNATURE_ATTENUATION_REPS if n_rep is None else int(n_rep)
    coordinates = cell.design_coordinates
    coordinates["dataset_key"] = cell.dataset_key + ("attenuation",)
    coordinates["T"] = simcfg.SIGNATURE_ATTENUATION_T
    coordinates["n_rep"] = n_rep
    longer = registry.Cell(**coordinates, statistics=cell.statistics)

    context = detection.build_context(longer)
    rows = []
    for replication in range(n_rep):
        row = detection.execute(longer, context, replication)[0]
        row.update({"status": "ok", "record_index": 0, "replication_id": replication})
        rows.append(row)

    frame = influence_path_data(longer, pd.DataFrame(rows), context, None)
    agreement = signature_agreement(frame)

    return {
        "attenuation_T": simcfg.SIGNATURE_ATTENUATION_T,
        "attenuation_n_replications": n_rep,
        "attenuation_regression_coefficient": agreement["regression_coefficient"],
        "attenuation_correlation": agreement["correlation"],
        "attenuation_root_mean_squared_difference": agreement[
            "root_mean_squared_difference"
        ],
    }


def build_influence_outputs(cells: list, paths: simio.RunPaths) -> dict:
    """Draw and write the influence-path figure data and the agreement table.

    Parameters
    ----------
    cells : list
        The registry.
    paths : RunPaths
        The run layout.

    Returns
    -------
    dict
        Mapping from output name to the frame behind it.
    """
    null_cell = next(
        (
            cell
            for cell in cells
            if cell.kind == "null_calibration"
            and (cell.m, cell.n, cell.T)
            == (simcfg.BASE_M, simcfg.BASE_N, simcfg.BASE_T)
            and cell.innovation == "isotropic"
            and cell.persistence == simcfg.BASE_PERSISTENCE
        ),
        None,
    )

    written: dict[str, pd.DataFrame] = {}
    agreements = []

    for cell in cells:
        if "influence_path" not in cell.statistics:
            continue

        records = simio.read_cell_records(paths, cell)
        if records.empty or "influence_projection" not in records:
            continue

        context = detection.build_context(cell)
        null_path = (
            null_realisation(cell, context, null_cell)
            if null_cell is not None
            else None
        )

        frame = influence_path_data(cell, records, context, null_path)
        if frame.empty:
            continue

        name = f"influence_path_{cell.block}_{cell.path}_{cell.magnitude}"
        simio.save_table(paths, name, frame, simcfg.TABLE_GROUP_MAIN)
        simio.save_figure(paths, name, draw_influence_path(frame, cell))
        written[name] = frame

        agreement = signature_agreement(frame)
        agreement.update(attenuation_at_longer_sample(cell))
        agreement["attenuation_shrinks"] = bool(
            abs(agreement["attenuation_regression_coefficient"] - 1.0)
            < abs(agreement["regression_coefficient"] - 1.0)
        )
        agreements.append(agreement)

    if agreements:
        table = pd.DataFrame(agreements)
        simio.save_table(
            paths,
            "influence_path_signature_agreement",
            table,
            simcfg.TABLE_GROUP_MAIN,
        )
        written["influence_path_signature_agreement"] = table

    return written
