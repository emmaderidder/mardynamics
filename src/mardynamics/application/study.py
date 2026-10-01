"""The empirical study orchestrator.

The order is fixed: the data are prepared before any fit, the static
reference precedes the constancy analysis, and the forecast design and
comparison precede the description of the fitted pooling estimates.

Every value comes from :mod:`mardynamics.application.emp_config`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import mardynamics.application.structural_change as constancy
import mardynamics.application.data as data_mod
import mardynamics.application.descriptions as descriptions
import mardynamics.application.emp_config as emp_config
import mardynamics.application.estimators as estimators
import mardynamics.application.evaluation as evaluation
import mardynamics.application.figures as figures
import mardynamics.application.forecasting as forecasting
import mardynamics.application.run_infrastructure as runinfra
import mardynamics.core.structural_tests as structural_tests


def _static_fit_with_telemetry(tensor: np.ndarray) -> tuple:
    """Fit the static MAR(1) and assemble its telemetry.

    Reuses :mod:`mardynamics.application.forecasting`'s convergence probe
    and stability telemetry helpers so the retrospective static fit carries
    the same estimator identity, convergence status, iteration count,
    stopping reason, spectral radius and stability-contraction flag as
    every forecast-loop static fit.

    Parameters
    ----------
    tensor : numpy.ndarray
        The tensor to fit on.

    Returns
    -------
    tuple
        ``((A, B), telemetry)``.
    """
    A, B, solver = estimators.static_fit(tensor, return_telemetry=True)
    iterations = int(solver["als_iterations"])
    converged, reason = forecasting._als_stopping(
        tensor,
        None,
        iterations,
        max_iter=emp_config.STATIC_MAX_ITER,
        tol=emp_config.STATIC_TOL,
        stopping_rule=emp_config.STATIC_STOPPING_RULE,
    )
    radius = forecasting._joint_spectral_radius(A, B)
    telemetry = {
        "estimator_identity": solver["estimator_identity"],
        "objective": "mar_lse_kronecker_projection_init",
        "n_iter": iterations,
        "converged": converged,
        "convergence_reason": reason,
        "criterion_before": float(solver["criterion_before"]),
        "criterion_after": float(solver["criterion_after"]),
        "joint_spectral_radius": radius,
        "stability_contracted": forecasting.inferred_contraction(radius),
    }
    return (A, B), telemetry


def run(mode: str, workers: int, data_dir: str = "data") -> dict:
    """Execute the complete empirical pipeline in one run mode.

    Parameters
    ----------
    mode : str
        ``"preview"`` or ``"full"``.
    workers : int
        Process workers, capped at MAX_WORKERS.
    data_dir : str, optional
        Directory holding the raw source exports.

    Returns
    -------
    dict
        The closing summary of the run.
    """
    runinfra.enforce_thread_limits()
    started = time.perf_counter()

    paths = runinfra.create_run(mode)
    failures = runinfra.FailureLog()
    print(f"[run   ] {paths.root}  hash={paths.hash}")

    # ---------------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------------
    prepared = data_mod.prepare(data_dir)
    # The retrospective analysis reads the full-sample scale. Every
    # forecast reads an origin-safe one. The two are not mixed, and no
    # fitted object crosses between them without being refitted on the
    # reading section's own scale.
    retrospective = prepared.tensors[emp_config.SECTION_PREPROCESSING["5.3_constancy"]]
    tensor = prepared.tensors[emp_config.SECTION_PREPROCESSING["5.4_test"]]
    periods = prepared.periods
    print(
        f"[data  ] {tensor.shape[0]} matrices, {tensor.shape[0] - 1} transitions, "
        f"{len(prepared.adf)} ADF diagnostics"
    )

    for version in emp_config.PREPROCESSOR_CUTOFFS:
        runinfra.save_json(
            paths,
            f"preprocessing_{version}",
            prepared.preprocessors[version].to_payload(),
        )
    runinfra.save_table(
        paths,
        "adf_diagnostics",
        prepared.adf,
        preprocessing=emp_config.SECTION_PREPROCESSING["5.1_adf"],
    )

    validation_targets = data_mod.forecast_origins(periods, "validation", mode)
    test_targets = data_mod.forecast_origins(periods, "test", mode)

    # ---------------------------------------------------------------
    # Primary retrospective analysis, full-sample scale
    # ---------------------------------------------------------------
    scale = emp_config.SECTION_PREPROCESSING["5.3_constancy"]
    static_pair, static_telemetry = _static_fit_with_telemetry(retrospective)
    quantities = structural_tests.chart_score_quantities(retrospective, *static_pair)

    static_table, static_summary = constancy.static_reference(
        retrospective, static_pair, estimator_telemetry=static_telemetry
    )
    runinfra.save_table(
        paths,
        "static_reference_fit",
        static_table,
        preprocessing=scale,
        retrospective_route="full_sample_primary",
    )

    global_result = constancy.global_constancy(
        retrospective, static_pair, quantities, mode, failures
    )
    variance_result = constancy.innovation_variance(retrospective, static_pair)
    variance_result["variance_constancy_joint_status"] = (
        constancy.variance_constancy_status(
            variance_result["p_value"], global_result["p_value"]
        )
    )
    runinfra.save_table(
        paths,
        "innovation_variance",
        pd.DataFrame([variance_result]),
        preprocessing=scale,
        retrospective_route="full_sample_primary",
    )
    runinfra.save_table(
        paths,
        "constancy_global",
        pd.DataFrame([global_result]),
        preprocessing=scale,
        retrospective_route="full_sample_primary",
    )

    blocks = constancy.block_attribution(
        retrospective, static_pair, quantities, global_result["p_value"]
    )
    runinfra.save_table(
        paths,
        "constancy_blocks",
        blocks,
        preprocessing=scale,
        retrospective_route="full_sample_primary",
    )

    paths_frame = constancy.influence_paths(quantities, periods)
    runinfra.save_table(
        paths,
        "influence_paths",
        paths_frame,
        preprocessing=scale,
        retrospective_route="full_sample_primary",
    )
    print(
        f"[struct] full_sample_primary global {global_result['statistic']:.4f} "
        f"(95% c.v. {global_result['critical_95']:.4f}, "
        f"p={global_result['p_value']:.4f}), bootstrap "
        f"p={global_result['bootstrap_p_value']:.4f} on "
        f"{global_result['bootstrap_draws']} draws"
    )

    # ---------------------------------------------------------------
    # Forecast tuning and test comparison
    # ---------------------------------------------------------------
    candidates = forecasting.candidate_specs()
    print(
        f"[fcast ] tuning {len(candidates)} candidates over "
        f"{len(validation_targets)} validation origins"
    )
    # Validation is fit and evaluated on the validation scale, so its
    # fit/eval mapping is the identity.
    validation_preprocessor = prepared.preprocessors["validation"]
    validation_records = forecasting.run_units(
        candidates,
        prepared.tensors["validation"],
        prepared.tensors["validation"],
        validation_preprocessor,
        validation_preprocessor,
        periods,
        validation_targets,
        workers,
    )
    tuning, selections = evaluation.select_tuning(
        validation_records, validation_targets
    )
    runinfra.save_table(
        paths, "validation_tuning", tuning, preprocessing="validation"
    )

    runinfra.write_checkpoint(paths, "selections", selections)
    for model, choice in selections.items():
        print(
            f"       selected {model}: w={choice['window']} r={choice['rank']} "
            f"msfe={choice['validation_msfe']:.5f}"
        )

    for record in validation_records[validation_records["status"] != "ok"].itertuples():
        failures.record(
            unit="validation_forecast",
            identity={"candidate": record.candidate, "period": record.period},
            reason=record.failure_reason,
            message=record.failure_message,
        )

    # If a model has no feasible candidate, the lookup below raises.
    selected_specs = [
        forecasting.ModelSpec(
            model=model,
            rank=selections[model]["rank"],
            window=selections[model]["window"],
        )
        for model in emp_config.MODELS
    ]
    # Test models remain fit on the test scale. Every test prediction is
    # converted to the validation evaluation scale by
    # forecasting.run_units before an error is computed, and the actual
    # is read directly from the validation-scale tensor: the
    # test-fitting tensor's own actual is not used.
    print(f"       test forecasts over {len(test_targets)} origins")
    test_records = forecasting.run_units(
        selected_specs,
        tensor,
        prepared.tensors["validation"],
        prepared.preprocessors["test"],
        validation_preprocessor,
        periods,
        test_targets,
        workers,
    )
    for record in test_records[test_records["status"] != "ok"].itertuples():
        failures.record(
            unit="test_forecast",
            identity={"model": record.model, "period": record.period},
            reason=record.failure_reason,
            message=record.failure_message,
        )

    common, dropped = evaluation.common_origins(test_records, test_targets)
    print(
        f"       common origins {len(common)} of {len(test_targets)}"
        + (f", dropped {dropped}" if dropped else "")
    )
    # Every loss, accuracy, DM and GW number below is computed from
    # evaluation-scale (validation) errors, per forecasting.run_units
    # above, even though the models producing them were fit on the test
    # scale. Stamped "validation" to state the scale the reported numbers
    # live on, with fitting_preprocessor and evaluation_preprocessor
    # carried per row in raw_forecast_test.csv.
    runinfra.save_table(
        paths,
        "raw_forecast_test",
        evaluation.raw_forecast_records(test_records, common),
        preprocessing="validation",
    )

    accuracy, relative = evaluation.accuracy_tables(
        test_records, test_targets, common, dropped
    )
    runinfra.save_table(paths, "forecast_accuracy", accuracy, preprocessing="validation")
    runinfra.save_table(paths, "relative_msfe", relative, preprocessing="validation")

    loss_path = evaluation.period_loss(test_records, common)
    runinfra.save_table(paths, "period_loss", loss_path, preprocessing="validation")

    by_indicator, by_country = evaluation.cellwise_losses(test_records, common)
    runinfra.save_table(
        paths, "loss_by_indicator", by_indicator, preprocessing="validation"
    )
    runinfra.save_table(
        paths, "loss_by_country", by_country, preprocessing="validation"
    )

    ms_records = test_records[test_records["model"] == "ms_full"]
    p_high = pd.Series(
        ms_records.set_index("period")["p_high_predicted"].astype(float)
    )
    regime_split = evaluation.predicted_regime_split(test_records, common, p_high)
    runinfra.save_table(
        paths, "loss_by_predicted_regime", regime_split, preprocessing="validation"
    )

    dm = evaluation.diebold_mariano(test_records, common)
    gw = evaluation.giacomini_white(test_records, common, p_high)
    runinfra.save_table(paths, "forecast_tests_dm", dm, preprocessing="validation")
    runinfra.save_table(paths, "forecast_tests_gw", gw, preprocessing="validation")

    # ---------------------------------------------------------------
    # Rolling path and regime structure
    # ---------------------------------------------------------------
    # The rolling path is compared against a static reference. The
    # full-sample reference lives on the full-sample scale, and a
    # canonical shape is not invariant to the per-indicator rescaling, so
    # differencing the two would compare objects on different scales.
    # This section uses the static forecast benchmark refitted on the
    # test-preprocessing scale, the scale every quantity here already
    # sits on.
    benchmark_pair = estimators.static_fit(
        prepared.tensors[emp_config.SECTION_55_BENCHMARK_VERSION]
    )
    rolling_window = selections["rolling_full"]["window"]
    rolling_records = test_records[test_records["model"] == "rolling_full"]
    rolling_path, rolling_increments = descriptions.rolling_description(
        rolling_records, rolling_window, benchmark_pair
    )
    runinfra.save_table(
        paths, "rolling_canonical_path", rolling_path, preprocessing="test"
    )
    runinfra.save_table(
        paths, "rolling_increments", rolling_increments, preprocessing="test"
    )

    full_sample_fits = {
        ("ms_full", "full_sample"): estimators.regime_fit(tensor, "full_sample")
    }
    reduced_rank = selections.get("ms_reduced", {}).get("rank")
    if reduced_rank is not None:
        full_sample_fits[("ms_reduced", "full_sample")] = estimators.regime_fit(
            tensor, f"full_sample_r{reduced_rank}", rank=reduced_rank
        )

    regime = descriptions.regime_description(
        full_sample_fits[("ms_full", "full_sample")], periods
    )
    for name in (
        "regime_structure",
        "transition_matrix",
        "regime_neff",
        "regime_probabilities_smoothed",
    ):
        runinfra.save_table(paths, name, regime[name], preprocessing="test")

    oos = descriptions.predicted_probabilities(ms_records)
    runinfra.save_table(paths, "regime_probabilities_oos", oos, preprocessing="test")

    initialisation = descriptions.initialisation_table(full_sample_fits)
    telemetry = descriptions.telemetry_table(full_sample_fits, test_records)
    runinfra.save_table(
        paths, "initialisation_diagnostics", initialisation, preprocessing="test"
    )
    runinfra.save_table(
        paths, "estimator_telemetry", telemetry, preprocessing="test"
    )

    rank_energy = descriptions.rank_energy_table(
        tuning, benchmark_pair, full_sample_fits
    )
    runinfra.save_table(
        paths, "rank_tuning_and_energy", rank_energy, preprocessing="test"
    )

    # ---------------------------------------------------------------
    # Manifest, text-ready outputs and the closing check
    # ---------------------------------------------------------------
    manifest = _manifest(
        prepared,
        static_summary,
        validation_targets,
        test_targets,
        common,
        dropped,
        test_records,
        failures,
        mode,
        regime["interpretation_conditions"],
        static_telemetry,
    )
    runinfra.save_json(paths, "empirical_design_manifest", manifest)

    figures.influence_paths(paths, paths_frame)
    figures.validation(paths, tuning)
    figures.period_loss(paths, loss_path)
    figures.rolling_path(paths, rolling_path)
    figures.rolling_increments(paths, rolling_increments)
    figures.regime_smoothed(paths, regime["regime_probabilities_smoothed"])
    figures.regime_oos(paths, oos)

    failure_frame = failures.frame()
    if not failure_frame.empty:
        runinfra.save_table(
            paths, "failure_telemetry", failure_frame, preprocessing="test"
        )
        # The failure log is telemetry, not a registered output.
        (paths.telemetry / "failures.csv").write_text(
            failure_frame.to_csv(index=False), encoding="utf-8"
        )
        (paths.tables / "failure_telemetry.csv").unlink(missing_ok=True)
    else:
        (paths.telemetry / "failures.csv").write_text(
            "unit,failure_reason,failure_message\n", encoding="utf-8"
        )

    elapsed = time.perf_counter() - started

    summary = {
        "run_id": paths.run_id,
        "elapsed_seconds": elapsed,
        "validation_units": int(len(validation_records)),
        "validation_failed": int((validation_records["status"] != "ok").sum()),
        "test_units": int(len(test_records)),
        "test_failed": int((test_records["status"] != "ok").sum()),
        "common_origins": len(common),
        "dropped_origins": dropped,
    }
    runinfra.mark_complete(paths, summary)
    return {"paths": paths, **summary}


def _manifest(
    prepared,
    static_summary: dict,
    validation_targets,
    test_targets,
    common,
    dropped,
    test_records,
    failures,
    mode: str,
    interpretation_conditions: dict,
    static_telemetry: dict,
) -> dict:
    """Assemble ``empirical_design_manifest.json``.

    Parameters
    ----------
    prepared : EmpiricalData
        The prepared data.
    static_summary : dict
        Scalar summary of the static reference.
    validation_targets, test_targets : list of str
        The origins of this run mode.
    common, dropped : list of str
        The common and dropped test origins.
    test_records : pandas.DataFrame
        Test-phase unit records.
    failures : FailureLog
        Failure accounting.
    mode : str
        Run mode.
    static_telemetry : dict
        Estimator telemetry of the full-sample primary static fit.

    Returns
    -------
    dict
        The manifest payload.
    """
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "panel": {
            "countries": list(emp_config.COUNTRIES),
            "country_labels": dict(emp_config.COUNTRY_LABELS),
            "indicators": list(emp_config.INDICATORS),
            "indicator_labels": dict(emp_config.INDICATOR_LABELS),
            "orientation": "indicators on rows, countries on columns",
            "frequency": "monthly",
            "source": "OECD Data Explorer",
        },
        "samples": {
            "raw": [emp_config.RAW_START, emp_config.RAW_END],
            "full": [emp_config.SAMPLE_START, emp_config.SAMPLE_END],
            "validation": [emp_config.VALIDATION_START, emp_config.VALIDATION_END],
            "test": [emp_config.TEST_START, emp_config.TEST_END],
            "preprocessor_cutoffs": dict(emp_config.PREPROCESSOR_CUTOFFS),
            # Which part of the study reads which scale. The retrospective
            # analysis reads the full-sample scale. Every forecast reads
            # an origin-safe one, so no forecast can see beyond its own
            # origin. The two scales are not differenced against each
            # other.
            "section_preprocessing": dict(emp_config.SECTION_PREPROCESSING),
            "section_55_benchmark_version": emp_config.SECTION_55_BENCHMARK_VERSION,
        },
        "counts": {
            "matrices": int(prepared.tensors["test"].shape[0]),
            "transitions": int(prepared.tensors["test"].shape[0] - 1),
            "expected_matrices": emp_config.EXPECTED_MATRICES,
            "expected_transitions": emp_config.EXPECTED_TRANSITIONS,
            "planned_test_origins": emp_config.PLANNED_TEST_ORIGINS,
            "canonical_dimension_k": emp_config.K_CANONICAL,
            "adf_diagnostics": int(len(prepared.adf)),
        },
        "provenance": prepared.provenance,
        "substitutions": prepared.substitutions,
        "static_reference": static_summary,
        "numerical_settings": runinfra.design_payload(mode),
        # Author-decided values with no fixed numeric standard. They
        # travel with the run so no reader mistakes them for a
        # methodology constant.
        "author_decisions": {
            "regime_interpretation_thresholds": dict(
                emp_config.REGIME_INTERPRETATION_THRESHOLDS
            ),
            "regime_interpretation_conditions": interpretation_conditions,
        },
        "simulation_benchmark": {
            "empirical_k_over_N": round(
                emp_config.SIMULATION_BENCHMARK["empirical_k_over_N"], 3
            ),
            "benchmark_cell": emp_config.SIMULATION_BENCHMARK["benchmark_cell"],
            "benchmark_k_over_N": round(
                emp_config.SIMULATION_BENCHMARK["benchmark_k_over_N"], 3
            ),
            "benchmark_size_at_5pct": emp_config.SIMULATION_BENCHMARK[
                "benchmark_size_at_5pct"
            ],
            "statement": emp_config.SIMULATION_BENCHMARK["statement"],
        },
        "origins": {
            "validation_used": validation_targets,
            "test_used": test_targets,
            "common_test_origins": common,
            "dropped_test_origins": dropped,
            "origin_stride": (
                emp_config.PREVIEW_ORIGIN_STRIDE if mode == "preview" else 1
            ),
        },
        "failure_accounting": {
            "planned_test_origins_this_mode": len(test_targets),
            "attempted": int(len(test_records)),
            "successful": int((test_records["status"] == "ok").sum()),
            "failed": int((test_records["status"] != "ok").sum()),
            "hard_failures": failures.records,
            "policy": (
                "a failed forecast is never replaced; if any model hard-fails at "
                "an origin, that origin leaves every comparative loss and "
                "forecast-test calculation for all six models"
            ),
        },
        # Every active MAR-LSE route (static full-rank for the
        # retrospective, validation and test fits, rolling,
        # reduced-rank and regime-switching M-steps, and the
        # fixed-regressor wild-bootstrap refit) runs through the one
        # canonical NumPy projection-initialised ALS kernel, at the
        # settings named below.
        "estimator": {
            "identity": estimators.ESTIMATOR_IDENTITY,
            "objective": "mar_lse_kronecker_projection_init",
            "static_settings": {
                "max_iter": emp_config.STATIC_MAX_ITER,
                "tol": emp_config.STATIC_TOL,
                "stopping_rule": emp_config.STATIC_STOPPING_RULE,
            },
            "als_settings": {
                "max_iter": emp_config.ALS_MAX_ITER,
                "tol": emp_config.ALS_TOL,
                "stopping_rule": emp_config.ALS_STOPPING_RULE,
            },
            "stability": {
                "enforce": emp_config.ENFORCE_STABILITY,
                "max_radius": emp_config.MAX_RADIUS,
            },
            "bootstrap": {
                "route": "same canonical NumPy kernel as the observed static "
                "statistic, at the static settings above",
                "draws": emp_config.wild_bootstrap_draws(mode),
                "seed": emp_config.WILD_BOOT_SEED,
            },
            "full_sample_primary_static_telemetry": static_telemetry,
        },
        # The retrospective route.
        "retrospective_routes": {
            "full_sample_primary": {
                "preprocessing_version": "full_sample",
                "cutoff": emp_config.PREPROCESSOR_CUTOFFS["full_sample"],
                "role": "primary retrospective stability analysis",
            },
        },
        # The one canonical fit-to-evaluation mapping.
        "fitting_to_evaluation_mapping": {
            "validation": {
                "fitting_preprocessor": "validation",
                "evaluation_preprocessor": "validation",
                "mapping": "identity",
            },
            "test": {
                "fitting_preprocessor": "test",
                "evaluation_preprocessor": "validation",
                "mapping": (
                    "mardynamics.application.data.map_scale: every test prediction is "
                    "unstandardised on the test scale, CPI-reseasonalised from "
                    "the test seasonal means, re-seasonalised and rescaled onto "
                    "the validation scale; the actual is read directly from the "
                    "validation-scale tensor"
                ),
            },
            "loss_definition": (
                "squared evaluation-scale error, summed over indicators and "
                "countries; every accuracy, DM and GW number in this run is "
                "computed from evaluation-scale (validation/P07) errors, "
                "reconstructible from raw_forecast_test.csv"
            ),
        },
        "source_provenance": runinfra.source_provenance(),
    }
