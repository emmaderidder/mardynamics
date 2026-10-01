"""Construct every designed object of a registry and verify it.

The design fails before any simulation runs. A change direction that
misses its global-signal normalisation, a path that reaches the
stability boundary or loses the sign chart, a non-separable departure
that is not tangent orthogonal, or a population whose designated
coordinate sits too close to zero are all design-construction failures,
not run-time surprises. This module builds all of them once, keeps
every realised number, and hands back the resulting tables.

It also owns the cached cross-check of the analytic signal metric. That
check simulates a long panel per population, so it is run once per
population and configuration and reused afterwards, including by a
resumed run.

A violation is reported with its population configuration, changing
block, path, departure size, failed condition, spectral radius range,
minimum sign coordinate and seed, and it stops the run.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import mardynamics.experiments.design as design
import mardynamics.experiments.seeds as seeds
import mardynamics.experiments.sim_config as simcfg
import mardynamics.experiments.sim_io as simio

# Cache of the population-metric cross-checks, kept outside any run
# directory and shared across runs.
DESIGN_CACHE_NAME = "_design_cache"
G_CHECK_FILE = "g_moment_checks.json"


def _population_config_hash(truth: design.CanonicalTruth) -> str:
    """Identity of a population and the constants its metric depends on.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.

    Returns
    -------
    str
        A short hash.
    """
    payload = {
        "m": truth.m,
        "n": truth.n,
        "persistence": truth.persistence,
        "configuration": truth.configuration,
        "attempt": truth.attempt_id,
        "master_seed": simcfg.SIM_MASTER_SEED,
        "sigma": simcfg.SIGMA,
        "burn_in": simcfg.BURN_IN,
        "n_obs": simcfg.G_MOMENT_CHECK_OBS,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def cached_moment_check(truth: design.CanonicalTruth, cache_root: Path) -> dict:
    """Cross-check the analytic signal metric, reusing a cached result.

    Parameters
    ----------
    truth : CanonicalTruth
        The population whose metric is checked.
    cache_root : pathlib.Path
        Directory holding the cache file.

    Returns
    -------
    dict
        The check result, with ``cached`` recording whether it was reused.

    Raises
    ------
    design.DesignConstructionError
        If the analytic metric misses the tolerance.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / G_CHECK_FILE
    key = _population_config_hash(truth)

    store = {}
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            store = {}

    if key in store:
        return {**store[key], "cached": True}

    result = design.moment_check_signal_metric(
        truth,
        simcfg.SIGMA,
        simcfg.G_MOMENT_CHECK_OBS,
        seeds.cell_rng(
            "g_moment_check",
            truth.m,
            truth.n,
            truth.persistence,
            truth.configuration,
        ),
    )
    if not result["passed"]:
        raise design.DesignConstructionError(
            f"cached_moment_check: the analytic signal metric for "
            f"({truth.m}, {truth.n}, {truth.persistence}, {truth.configuration!r}) "
            f"differs from a {result['n_obs']}-observation moment simulation by "
            f"{result['relative_difference']:.4f}, above {result['tolerance']}."
        )

    store[key] = result
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temporary.replace(path)

    return {**result, "cached": False}


def population_keys(cells: list) -> list[tuple]:
    """Distinct population identities the registry needs.

    Parameters
    ----------
    cells : list
        The registry.

    Returns
    -------
    list of tuple
        ``(m, n, persistence, configuration)`` keys, in first-seen order.
    """
    keys = []
    for cell in cells:
        key = (cell.m, cell.n, cell.persistence, cell.configuration)
        if key not in keys:
            keys.append(key)
    return keys


def path_designs(cells: list) -> list[tuple]:
    """Every designed path that must be verified.

    Two sources are unioned: the paths the registry actually declares,
    and the complete cross product of changing block, scalar path and
    departure size at each population configuration and the reference
    sample length, which costs nothing because path construction is
    analytic.

    Parameters
    ----------
    cells : list
        The registry.

    Returns
    -------
    list of tuple
        ``(m, n, persistence, configuration, block, path, magnitude, T)``.
    """
    designs: list[tuple] = []

    for cell in cells:
        if cell.block is None or cell.path is None or cell.magnitude is None:
            continue
        signature = (
            cell.m,
            cell.n,
            cell.persistence,
            cell.configuration,
            cell.block,
            cell.path,
            cell.magnitude,
            cell.T,
        )
        if signature not in designs:
            designs.append(signature)

    for configuration in simcfg.POPULATION_CONFIGURATIONS:
        for block, path, magnitude in product(
            simcfg.PLANTED_BLOCKS, simcfg.PATHS, simcfg.MAGNITUDES
        ):
            signature = (
                simcfg.BASE_M,
                simcfg.BASE_N,
                simcfg.BASE_PERSISTENCE,
                configuration,
                block,
                path,
                magnitude,
                simcfg.BASE_T,
            )
            if signature not in designs:
                designs.append(signature)

    return designs


def audit(cells: list, output_dir: str | Path, run_moment_check: bool = True) -> dict:
    """Construct and verify every designed object of a registry.

    Parameters
    ----------
    cells : list
        The registry.
    output_dir : str or pathlib.Path
        Project output root; the design cache lives below it.
    run_moment_check : bool, default True
        Whether to run (or reuse) the population-metric cross-check.

    Returns
    -------
    dict
        The tables and the constructed objects, keyed by table name.

    Raises
    ------
    design.DesignConstructionError
        On the first inadmissible designed object.
    design.PathVerificationError
        On the first path that leaves the maintained simulation design.
    """
    # Reads the subdirectory from sim_io instead of repeating the
    # literal: two copies of the same path constant could silently
    # desync if it were ever renamed, leaving this cache written
    # somewhere the run cannot find it.
    cache_root = Path(output_dir) / simio.SIM_SUBDIR / DESIGN_CACHE_NAME

    populations: dict[tuple, design.CanonicalTruth] = {}
    metrics_by_key: dict[tuple, tuple] = {}
    directions: dict[tuple, design.PlantedDirection] = {}

    population_rows: list[dict] = []
    direction_rows: list[dict] = []
    signal_rows: list[dict] = []
    rejected_rows: list[dict] = []
    verification_rows: list[dict] = []
    departure_rows: list[dict] = []
    unrestricted_rows: list[dict] = []

    for key in population_keys(cells):
        m, n, persistence, configuration = key
        truth = design.build_truth(m, n, persistence, configuration)
        G = design.signal_metric(truth, simcfg.SIGMA)
        V = design.parameter_covariance(truth, simcfg.SIGMA)
        H = design.population_information(truth, simcfg.SIGMA)
        Omega = design.population_score_covariance(truth, simcfg.SIGMA)

        populations[key] = truth
        metrics_by_key[key] = (G, V)

        for seed_key, condition in truth.rejected_draws:
            rejected_rows.append(
                {
                    "m": m,
                    "n": n,
                    "persistence": persistence,
                    "configuration": configuration,
                    "seed_key": "|".join(str(part) for part in seed_key),
                    "failed_condition": condition,
                }
            )

        row = {
            "m": m,
            "n": n,
            "persistence": persistence,
            "configuration": configuration,
            "is_fixed_reference_population": (
                configuration == simcfg.REFERENCE_CONFIGURATION
            ),
            "attempt_id": truth.attempt_id,
            "n_rejected_draws": len(truth.rejected_draws),
            "seed_key": "|".join(str(part) for part in truth.seed_key),
            "sign_coordinate": truth.sign_coordinate,
            "rho_A_bar": float(np.max(np.abs(np.linalg.eigvals(truth.A_bar)))),
            "rho_B_breve": float(np.max(np.abs(np.linalg.eigvals(truth.B_breve)))),
            "gamma_0": truth.gamma,
            "kappa_0": truth.kappa,
            "rho_Phi_0": float(np.max(np.abs(np.linalg.eigvals(truth.Phi)))),
            "k": truth.k,
            "k_A": simcfg.block_dimension("row", m, n),
            "k_B": simcfg.block_dimension("column", m, n),
            "k_gamma": simcfg.block_dimension("amplitude", m, n),
            "min_eigenvalue_H": float(np.min(np.linalg.eigvalsh(H))),
            "cond_H": float(np.linalg.cond(H)),
            "min_eigenvalue_Omega": float(np.min(np.linalg.eigvalsh(Omega))),
            "cond_Omega": float(np.linalg.cond(Omega)),
            "A_bar_vec": ";".join(
                f"{value:.12e}" for value in truth.A_bar.flatten(order="F")
            ),
            "B_breve_vec": ";".join(
                f"{value:.12e}" for value in truth.B_breve.flatten(order="F")
            ),
        }
        if run_moment_check:
            check = cached_moment_check(truth, cache_root)
            row["g_moment_relative_difference"] = check["relative_difference"]
            row["g_moment_tolerance"] = check["tolerance"]
            row["g_moment_n_obs"] = check["n_obs"]
            row["g_moment_cached"] = check["cached"]
        population_rows.append(row)

        for block in simcfg.PLANTED_BLOCKS:
            direction = design.planted_direction(truth, block, G, V)
            directions[key + (block,)] = direction
            R = design.selection_matrix(truth, block)

            direction_rows.append(
                {
                    "m": m,
                    "n": n,
                    "persistence": persistence,
                    "configuration": configuration,
                    "block": block,
                    "k_block": direction.k_block,
                    "k_full": direction.k_full,
                    "seed_key": "|".join(str(p) for p in direction.seed_key),
                    "block_coordinates": ";".join(
                        f"{value:.12e}" for value in direction.block_coordinates
                    ),
                    "ambient_tangent_norm": float(
                        np.linalg.norm(direction.ambient_tangent)
                    ),
                    "tangent_orthogonality_error": direction.tangent_orthogonality,
                }
            )

            V_block = R @ V @ R.T
            signal_rows.append(
                {
                    "m": m,
                    "n": n,
                    "persistence": persistence,
                    "configuration": configuration,
                    "block": block,
                    # The two measures are reported side by side and are not
                    # the same quantity: equal global signal does not imply
                    # equal blockwise signal.
                    "global_signal_v_G_v": direction.global_signal,
                    "global_signal_deviation_from_one": (
                        direction.global_signal_deviation
                    ),
                    "blockwise_signal_lambda": direction.blockwise_signal,
                    "block_dimension_k_block": direction.k_block,
                    "dim_v_embedded": direction.k_full,
                    "dim_v_block": direction.k_block,
                    "dim_R_rows": int(R.shape[0]),
                    "dim_R_cols": int(R.shape[1]),
                    "dim_G": int(G.shape[0]),
                    "dim_V_block": int(V_block.shape[0]),
                    "tolerance": simcfg.G_NORMALISATION_TOLERANCE,
                }
            )

    for signature in path_designs(cells):
        m, n, persistence, configuration, block, path, magnitude, T = signature
        key = (m, n, persistence, configuration)
        if key not in populations:
            truth = design.build_truth(m, n, persistence, configuration)
            G = design.signal_metric(truth, simcfg.SIGMA)
            V = design.parameter_covariance(truth, simcfg.SIGMA)
            populations[key] = truth
            metrics_by_key[key] = (G, V)
        truth = populations[key]
        G, V = metrics_by_key[key]

        direction = directions.get(key + (block,))
        if direction is None:
            direction = design.planted_direction(truth, block, G, V)
            directions[key + (block,)] = direction

        built = design.build_parameter_path(truth, direction, path, magnitude, T)
        record = design.verify_parameter_path(truth, direction, built, strict=True)
        verification_rows.append(record)

        departure_rows.append(
            {
                "configuration": configuration,
                "m": m,
                "n": n,
                "T": T,
                "N": simcfg.score_observations(T),
                "block": block,
                "path": path,
                "magnitude": magnitude,
                "is_local_departure": magnitude in simcfg.LOCAL_MAGNITUDE_SCALARS,
                "departure_scalar": built.scalar,
                "local_scalar": (
                    np.nan if built.local_scalar is None else built.local_scalar
                ),
                "equivalent_local_scalar": built.equivalent_local_scalar,
                "integrated_squared_cumulative_demeaned_signature": (
                    built.signature_scale
                ),
                "noncentrality": built.noncentrality,
                "max_geodesic_displacement": record["max_geodesic_displacement"],
                "max_frobenius_displacement": record["max_frobenius_displacement"],
                "max_abs_log_magnitude_displacement": record[
                    "max_abs_log_magnitude_displacement"
                ],
                "blockwise_signal_lambda": direction.blockwise_signal,
                "global_signal_v_G_v": direction.global_signal,
            }
        )

    for key in {
        (cell.m, cell.n, cell.persistence, cell.configuration)
        for cell in cells
        if cell.kind == "matrix_restriction"
    }:
        truth = populations[key]
        G, V = metrics_by_key[key]
        departure = design.orthogonal_departure(truth, simcfg.SIGMA)
        induced = design.induced_separable_direction(
            truth, directions[key + ("row",)], simcfg.SIGMA
        )
        unrestricted_rows.append(
            {
                "m": key[0],
                "n": key[1],
                "persistence": key[2],
                "configuration": key[3],
                "k": truth.k,
                "K": int((key[0] * key[1]) ** 2),
                "separable_unrestricted_signal": induced.unrestricted_signal,
                "separable_signal_deviation": induced.deviation,
                "separable_rescaled": False,
                "non_separable_unrestricted_signal": departure.unrestricted_signal,
                "tangent_orthogonality_error": departure.orthogonality,
                "tangent_orthogonality_tolerance": (
                    simcfg.G_PERP_ORTHOGONALITY_TOLERANCE
                ),
                "jacobian_rank": departure.rank_feasibility["jacobian_rank"],
                "complement_dimension": departure.rank_feasibility[
                    "complement_dimension"
                ],
                "seed_key": "|".join(str(p) for p in departure.seed_key),
            }
        )

    return {
        "populations": populations,
        "metrics": metrics_by_key,
        "directions": directions,
        "population_matrices": pd.DataFrame(population_rows),
        "change_directions": pd.DataFrame(direction_rows),
        "signal_measures": pd.DataFrame(signal_rows),
        "rejected_population_draws": pd.DataFrame(
            rejected_rows,
            columns=[
                "m",
                "n",
                "persistence",
                "configuration",
                "seed_key",
                "failed_condition",
            ],
        ),
        "path_verification": pd.DataFrame(verification_rows),
        "departure_size_summary": pd.DataFrame(departure_rows),
        "unrestricted_directions": pd.DataFrame(unrestricted_rows),
    }


def seed_registry(cells: list, result: dict) -> pd.DataFrame:
    """Every seed identity of the registered cells, with its spawn key.

    Four roles are separated and not mixed: the population seeds that
    draw a canonical parameter, the direction seeds that draw a change
    direction, the replication seeds that draw a dataset, and the
    bootstrap and start sub-stream seeds that draw everything which must
    be independent of the data. Two rows with the same spawn key would
    be two draws sharing a stream, which is why the spawn key itself is
    recorded.

    The scope is the registry. One diagnostic outside it also draws
    data: the longer-sample signature attenuation of
    :func:`mardynamics.experiments.figures.attenuation_at_longer_sample`
    derives a stream ``dataset_key + ("attenuation",)`` from the
    influence-path cell it diagnoses. It is a sub-stream of a registered
    dataset key and cannot collide with any row here, but it is not
    itself a row, and the disjointness check run over this table does
    not range over it.

    Parameters
    ----------
    cells : list
        The registry.
    result : dict
        The audit result, for the populations and directions it constructed.

    Returns
    -------
    pandas.DataFrame
        One row per seed identity.
    """
    rows = []

    def _add(role, key, description, n_streams=1):
        rows.append(
            {
                "role": role,
                "seed_key": "|".join(str(part) for part in key),
                "spawn_key": "-".join(
                    str(word) for word in seeds.seed_key_words(*key)
                ),
                "description": description,
                "n_streams": int(n_streams),
                "master_seed": simcfg.SIM_MASTER_SEED,
            }
        )

    for key, truth in result["populations"].items():
        _add("population", truth.seed_key, f"accepted population {key}")
        for seed_key, condition in truth.rejected_draws:
            _add(
                "population_rejected",
                seed_key,
                f"rejected population {key}: {condition}",
            )

    for key, direction in result["directions"].items():
        if direction.seed_key:
            _add("direction", direction.seed_key, f"change direction {key}")

    for _, row in result["unrestricted_directions"].iterrows():
        _add(
            "direction",
            tuple(row["seed_key"].split("|")),
            "non-separable transition direction",
        )

    for cell in cells:
        _add(
            "replication",
            tuple(cell.dataset_key),
            f"datasets of {cell.cell_id}",
            n_streams=cell.n_rep,
        )
        if cell.kind == "wild_bootstrap":
            _add(
                "bootstrap",
                tuple(cell.dataset_key) + ("wild_bootstrap",),
                f"bootstrap draws of {cell.cell_id}",
                n_streams=cell.n_rep,
            )
        if cell.kind == "multiple_start":
            for start_id in range(1, 1 + int(cell.extra["n_perturbed_starts"])):
                _add(
                    "start",
                    tuple(cell.dataset_key) + (f"start_{start_id}",),
                    f"perturbed start {start_id} of {cell.cell_id}",
                    n_streams=cell.n_rep,
                )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["role", "seed_key"])
    return frame.reset_index(drop=True)


def seed_streams_are_disjoint(registry: pd.DataFrame) -> dict:
    """Check that no two distinct seed identities share a stream.

    Streams are derived from a cryptographic digest of the draw's
    meaning, not from an integer range, so "ranges do not overlap" is
    the statement that two different identities do not produce the same
    spawn key.

    Parameters
    ----------
    registry : pandas.DataFrame
        The seed registry.

    Returns
    -------
    dict
        The number of identities, the number of distinct spawn keys and the
        verdict.
    """
    n_identities = int(registry["seed_key"].nunique())
    n_spawn_keys = int(registry["spawn_key"].nunique())
    return {
        "n_seed_identities": n_identities,
        "n_distinct_spawn_keys": n_spawn_keys,
        "streams_disjoint": bool(n_identities == n_spawn_keys),
    }


def summarise(result: dict) -> str:
    """Render a short human-readable summary of an audit.

    Parameters
    ----------
    result : dict
        The audit result.

    Returns
    -------
    str
        The summary.
    """
    signals = result["signal_measures"]
    verification = result["path_verification"]

    lines = [
        f"populations {len(result['population_matrices'])} | "
        f"directions {len(result['change_directions'])} | "
        f"verified paths {len(verification)} | "
        f"rejected draws {len(result['rejected_population_draws'])}",
        f"max |v'Gv - 1| = {signals['global_signal_deviation_from_one'].max():.3e} "
        f"(tolerance {simcfg.G_NORMALISATION_TOLERANCE:g})",
        "blockwise signal range = "
        f"[{signals['blockwise_signal_lambda'].min():.4f}, "
        f"{signals['blockwise_signal_lambda'].max():.4f}] "
        "(global signal is one everywhere; the two are different quantities)",
    ]
    if not verification.empty:
        lines.append(
            f"max rho(Phi_t) over verified paths = "
            f"{verification['max_spectral_radius'].max():.4f} "
            f"(limit {simcfg.MAX_RADIUS}); "
            f"min sign coordinate = {verification['min_sign_coordinate'].min():.4f} "
            f"(margin {design.PATH_MIN_SIGN_COORDINATE:g})"
        )
    if "g_moment_relative_difference" in result["population_matrices"]:
        checks = result["population_matrices"]["g_moment_relative_difference"].dropna()
        if not checks.empty:
            lines.append(
                f"max relative signal-metric moment-check difference = "
                f"{checks.max():.4f} (tolerance {simcfg.G_MOMENT_CHECK_TOLERANCE})"
            )
    return "\n".join(lines)
