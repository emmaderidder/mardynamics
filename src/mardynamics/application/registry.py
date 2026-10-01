"""The typed design registry and output registry of the empirical study.

Describes the complete set of outputs: every required analysis, model,
tuning cell, statistic, path, telemetry field, table and figure.

Every registry item carries its section, empirical question, model,
sample, preprocessing version, tuning coordinates, quantity, required
machine-readable files, required text-ready outputs,
interpretation-status field and run-mode membership. Every physical file
carries its required schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mardynamics.application.emp_config as emp_config

# The three questions the empirical study answers.
QUESTIONS = {
    "Q1": "Is the identified canonical MAR(1) parameter constant over time?",
    "Q2": "If the global null is rejected, in which canonical block is the "
    "first-order evidence concentrated?",
    "Q3": "Do prespecified pooling and rank-regularisation rules improve "
    "genuine out-of-sample forecasts relative to the static full-rank "
    "benchmark?",
    "Q0": "Design, data and numerical-reliability record. Answers is no "
    "substantive question on its own.",
}


@dataclass(frozen=True)
class RegistryItem:
    """One analysis and the outputs it must produce.

    Attributes
    ----------
    item_id : str
        Stable identity of the item.
    section : str
        Section of the study this item belongs to.
    question : str
        Key into :data:`QUESTIONS`.
    model : str
        Model or estimator the item describes.
    sample : str
        Sample the item is computed on.
    preprocessing : str
        Preprocessing version used.
    tuning : tuple
        Tuning coordinates, empty when the item has none.
    quantity : str
        The statistic or descriptive quantity.
    files : tuple
        Required machine-readable outputs.
    figures : tuple
        Required figure sources.
    interpretation_status : str
        Name of the interpretation-status field, or an empty string.
    modes : tuple
        Run modes the item belongs to.
    """

    item_id: str
    section: str
    question: str
    model: str
    sample: str
    preprocessing: str
    quantity: str
    files: tuple = ()
    figures: tuple = ()
    tuning: tuple = ()
    interpretation_status: str = ""
    modes: tuple = ("preview", "full")


# The complete design registry, in the order of the study's sections.
REGISTRY_ITEMS: tuple[RegistryItem, ...] = (
    # ----- Panel, preprocessing and stationarity ------------------------
    RegistryItem(
        item_id="5.1.1_panel",
        section="5.1.1",
        question="Q0",
        model="none",
        sample="full",
        preprocessing="test",
        quantity="panel definition, sample counts, provenance and the "
        "approved source substitution",
        files=("empirical_design_manifest.json",),
    ),
    RegistryItem(
        item_id="5.1.3_preprocessors",
        section="5.1.3",
        question="Q0",
        model="none",
        sample="full, validation and test",
        preprocessing="all three",
        quantity="the three preprocessing parameter sets: a full-sample one "
        "for the retrospective analysis, and the two origin-safe ones the "
        "forecasts require",
        files=(
            "preprocessing_full_sample.json",
            "preprocessing_validation.json",
            "preprocessing_test.json",
        ),
    ),
    RegistryItem(
        item_id="5.1.2_adf",
        section="5.1.2",
        question="Q0",
        model="none",
        sample="full",
        preprocessing="test",
        quantity="25 ADF diagnostics, constant and no trend, Schwert maximum "
        "lag 17, AIC selection, MacKinnon p-values",
        files=("adf_diagnostics.csv",),
    ),
    # ----- Static reference and innovation variance ---------------------
    RegistryItem(
        item_id="5.2.1_static_reference",
        section="5.2.1",
        question="Q1",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="canonical row shape, column shape, amplitude, Kronecker "
        "transition operator and joint spectral radius",
        files=("static_reference_fit.csv",),
        interpretation_status="reference_description",
    ),
    RegistryItem(
        item_id="5.2.2_innovation_variance",
        section="5.2.2",
        question="Q1",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="auxiliary fluctuation statistic on the squared Frobenius "
        "innovation norm, Bartlett long-run variance",
        files=("innovation_variance.csv",),
        interpretation_status="variance_constancy_joint_status",
    ),
    # ----- Constancy tests -----------------------------------------------
    RegistryItem(
        item_id="5.3.1_global_nyblom",
        section="5.3.1",
        question="Q1",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="global canonical Nyblom-Hansen statistic, critical values "
        "and simulated-tail p-value at k=49 and N=415",
        files=("constancy_global.csv",),
        interpretation_status="constancy_decision_5pct",
    ),
    RegistryItem(
        item_id="5.3.2_wild_bootstrap",
        section="5.3.2",
        question="Q1",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="fixed-regressor Rademacher wild-bootstrap calibration of the "
        "same global statistic",
        files=("constancy_global.csv",),
        interpretation_status="bootstrap_agreement_status",
    ),
    RegistryItem(
        item_id="5.3.3_var_feasibility",
        section="5.3.3",
        question="Q1",
        model="unrestricted_var",
        sample="full",
        preprocessing="test",
        quantity="concise dimension comparison K=625 against score-covariance "
        "rank at most T-2=414; the statistic is refused analytically",
        files=("constancy_global.csv",),
        interpretation_status="var_feasibility_status",
    ),
    RegistryItem(
        item_id="5.3.6_simulation_benchmark",
        section="5.3.6",
        question="Q1",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="k/N and the one controlled simulation-cell comparison",
        files=("constancy_global.csv", "empirical_design_manifest.json"),
        interpretation_status="simulation_benchmark_statement",
    ),
    RegistryItem(
        item_id="5.3.4_blocks",
        section="5.3.4",
        question="Q2",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="three influence-based block statistics with Holm adjustment "
        "over exactly three p-values",
        files=("constancy_blocks.csv",),
        interpretation_status="attribution_status",
    ),
    RegistryItem(
        item_id="5.3.5_influence_paths",
        section="5.3.5",
        question="Q2",
        model="static_full",
        sample="full",
        preprocessing="test",
        quantity="four coordinate-invariant influence paths",
        files=("influence_paths.csv",),
        figures=("fig_influence_paths",),
        interpretation_status="path_description_only",
    ),
    # ----- Forecast tuning and accuracy -----------------------------------
    RegistryItem(
        item_id="5.4.2_validation_tuning",
        section="5.4.2",
        question="Q3",
        model="all six",
        sample="validation",
        preprocessing="validation",
        quantity="complete recursive one-step validation MSFE for every "
        "candidate, and the selections",
        files=("validation_tuning.csv",),
        figures=("fig_validation",),
        tuning=("rank", "window"),
        interpretation_status="selection_rule",
    ),
    RegistryItem(
        item_id="5.4.4_accuracy",
        section="5.4.4",
        question="Q3",
        model="all six",
        sample="test",
        preprocessing="test",
        quantity="mean, median and 90th percentile monthly loss and relative "
        "MSFE against the static full-rank benchmark",
        files=("forecast_accuracy.csv", "relative_msfe.csv"),
        interpretation_status="common_origin_status",
    ),
    RegistryItem(
        item_id="5.4.4_period_loss",
        section="5.4.4",
        question="Q3",
        model="all six",
        sample="test",
        preprocessing="test",
        quantity="monthly loss path on a logarithmic scale",
        files=("period_loss.csv",),
        figures=("fig_period_loss",),
    ),
    RegistryItem(
        item_id="5.4.4_loss_by_indicator",
        section="5.4.4",
        question="Q3",
        model="all six",
        sample="test",
        preprocessing="test",
        quantity="mean squared error aggregated by indicator",
        files=("loss_by_indicator.csv",),
    ),
    RegistryItem(
        item_id="5.4.4_loss_by_country",
        section="5.4.4",
        question="Q3",
        model="all six",
        sample="test",
        preprocessing="test",
        quantity="mean squared error aggregated by country",
        files=("loss_by_country.csv",),
    ),
    RegistryItem(
        item_id="5.4.4_predicted_regime_split",
        section="5.4.4",
        question="Q3",
        model="all six",
        sample="test",
        preprocessing="test",
        quantity="descriptive loss split by the ex-ante predicted "
        "high-variance probability of the full-rank MS-MAR",
        files=("loss_by_predicted_regime.csv",),
        interpretation_status="descriptive_not_a_test",
    ),
    RegistryItem(
        item_id="5.4.5_diebold_mariano",
        section="5.4.5",
        question="Q3",
        model="five challengers",
        sample="test",
        preprocessing="test",
        quantity="two-sided DM with HAC lag 3, Harvey-Leybourne-Newbold "
        "correction and Holm adjustment over five challengers",
        files=("forecast_tests_dm.csv",),
        interpretation_status="dm_interpretation",
    ),
    RegistryItem(
        item_id="5.4.6_giacomini_white",
        section="5.4.6",
        question="Q3",
        model="five challengers",
        sample="test",
        preprocessing="test",
        quantity="conditional comparison on the continuous ex-ante "
        "high-variance probability with HAC lag 3 and Holm adjustment",
        files=("forecast_tests_gw.csv",),
        interpretation_status="generated_state_caveats",
    ),
    # ----- Rolling path and regime structure ------------------------------
    RegistryItem(
        item_id="5.5.1_rolling_path",
        section="5.5.1",
        question="Q3",
        model="rolling_full",
        sample="test",
        preprocessing="test",
        quantity="canonical row-shape, column-shape and amplitude path with "
        "sequential sign alignment and canonical increments",
        files=("rolling_canonical_path.csv", "rolling_increments.csv"),
        figures=("fig_rolling_path", "fig_rolling_increments"),
        tuning=("window",),
        interpretation_status="forecast_selected_pooling_rule",
    ),
    RegistryItem(
        item_id="5.5.2_regime_structure",
        section="5.5.2",
        question="Q3",
        model="ms_full",
        sample="full",
        preprocessing="test",
        quantity="per-regime canonical triple, joint spectral radius, "
        "innovation variance, occupancy, effective sample size, "
        "transition matrix and expected durations",
        files=("regime_structure.csv", "transition_matrix.csv", "regime_neff.csv"),
        interpretation_status="regime_interpretation_status",
    ),
    RegistryItem(
        item_id="5.5.2_regime_probabilities",
        section="5.5.2",
        question="Q3",
        model="ms_full",
        sample="full and test",
        preprocessing="test",
        quantity="full-sample smoothed probabilities for retrospective "
        "description and ex-ante predicted probabilities over the test period",
        files=("regime_probabilities_smoothed.csv", "regime_probabilities_oos.csv"),
        figures=("fig_regime_smoothed", "fig_regime_oos"),
        interpretation_status="smoothed_never_used_in_forecasts",
    ),
    RegistryItem(
        item_id="5.5.2_telemetry",
        section="5.5.2",
        question="Q0",
        model="ms_full and ms_reduced",
        sample="full and test",
        preprocessing="test",
        quantity="all eight final likelihoods, retained start, convergence "
        "reason, GEM acceptances and rollbacks, likelihood monotonicity",
        files=("initialisation_diagnostics.csv", "estimator_telemetry.csv"),
        interpretation_status="numerical_reliability_status",
    ),
    RegistryItem(
        item_id="5.5.3_rank_role",
        section="5.5.3",
        question="Q3",
        model="static_reduced, rolling_reduced and ms_reduced",
        sample="validation and full",
        preprocessing="test",
        quantity="selected ranks, canonical parameter count, singular-value "
        "energy and regime effective sample sizes",
        files=("rank_tuning_and_energy.csv",),
        tuning=("rank",),
        interpretation_status="regularisation_not_true_rank",
    ),
    # ----- The raw forecast artifact ---------------------------------------
    # Scoped to the test phase (the six selected models), which is what
    # every DM, GW, accuracy, period-loss, cellwise and regime-split output
    # downstream is computed from. Validation candidate MSFEs are already
    # reconstructible from the scalar validation_tuning.csv.
    RegistryItem(
        item_id="final_decision_7_raw_forecast_test",
        section="final_decision_7",
        question="Q3",
        model="all six",
        sample="test",
        preprocessing="test fit mapped to validation evaluation scale",
        quantity="reconstructible long-format forecast record: prediction on "
        "the test-fitting scale, prediction mapped to and actual read from "
        "the validation evaluation scale, the evaluation-scale error and its "
        "squared contribution, common-origin eligibility and fit status, per "
        "model, origin, indicator and country",
        files=("raw_forecast_test.csv",),
    ),
)


# The 50 canonical shape entry columns of ``rolling_canonical_path.csv``.
_SHAPE_ENTRY_COLUMNS: tuple[str, ...] = tuple(
    f"A_bar_{predicted}_{predictor}"
    for predicted in emp_config.INDICATORS
    for predictor in emp_config.INDICATORS
) + tuple(
    f"B_breve_{predicted}_{predictor}"
    for predicted in emp_config.COUNTRIES
    for predictor in emp_config.COUNTRIES
)


# Required columns of every registered machine-readable table. The
# stamping columns are added by the run infrastructure and are checked
# separately.
TABLE_SCHEMAS: dict[str, tuple] = {
    "adf_diagnostics.csv": (
        "indicator",
        "country",
        "adf_statistic",
        "p_value",
        "selected_lag",
        "max_lag",
        "effective_observations",
        "regression",
        "lag_selection",
    ),
    "static_reference_fit.csv": (
        "quantity",
        "row",
        "column",
        "value",
    ),
    "innovation_variance.csv": (
        "statistic",
        "p_value",
        "bandwidth",
        "decision_1pct",
        "decision_5pct",
        "decision_10pct",
        "variance_constancy_joint_status",
    ),
    "constancy_global.csv": (
        "statistic_name",
        "statistic",
        "k",
        "n_score_observations",
        "critical_90",
        "critical_95",
        "critical_99",
        "p_value",
        "decision_1pct",
        "decision_5pct",
        "decision_10pct",
        "bootstrap_draws",
        "wild_bootstrap_seed",
        "bootstrap_p_value",
        "bootstrap_agreement_status",
        "var_parameter_directions",
        "var_score_rank_bound",
        "var_feasibility_status",
        "k_over_n",
        "benchmark_cell",
        "benchmark_k_over_n",
        "benchmark_size_at_5pct",
        "simulation_benchmark_statement",
        "constancy_decision_5pct",
    ),
    "constancy_blocks.csv": (
        "block",
        "block_dimension",
        "statistic",
        "p_value",
        "holm_p_value",
        "critical_90",
        "critical_95",
        "critical_99",
        "decision_5pct",
        "holm_decision_5pct",
        "attribution_status",
    ),
    "influence_paths.csv": (
        "period",
        "observation_index",
        "path",
        "value",
    ),
    "validation_tuning.csv": (
        "model",
        "rank",
        "window",
        "n_origins",
        "n_failed",
        "feasible",
        "validation_msfe",
        "selected",
        "selection_rule",
    ),
    "forecast_accuracy.csv": (
        "model",
        "n_planned",
        "n_attempted",
        "n_successful",
        "n_failed",
        "n_common_origins",
        "mean_loss",
        "median_loss",
        "p90_loss",
        "common_origin_status",
    ),
    "relative_msfe.csv": ("model", "mean_loss", "benchmark_mean_loss", "relative_msfe"),
    "period_loss.csv": ("period", "model", "loss", "log_loss"),
    "loss_by_indicator.csv": ("model", "indicator", "mean_squared_error"),
    "loss_by_country.csv": ("model", "country", "mean_squared_error"),
    "loss_by_predicted_regime.csv": (
        "model",
        "predicted_regime_group",
        "n_origins",
        "mean_loss",
        "descriptive_not_a_test",
    ),
    "forecast_tests_dm.csv": (
        "model",
        "mean_loss_differential",
        "hac_lag",
        "dm_statistic",
        "hln_statistic",
        "p_value",
        "holm_p_value",
        "n_common_origins",
        "dm_interpretation",
    ),
    "forecast_tests_gw.csv": (
        "model",
        "joint_wald_statistic",
        "joint_p_value",
        "holm_joint_p_value",
        "state_coefficient",
        "state_p_value",
        "hac_lag",
        "n_common_origins",
        "generated_state_caveats",
    ),
    "rolling_canonical_path.csv": (
        "period",
        "window",
        "gamma",
        "joint_spectral_radius",
        # The reference is the benchmark refitted on the test-preprocessing
        # scale.
        "row_shape_distance_to_test_scale_benchmark",
        "column_shape_distance_to_test_scale_benchmark",
        "distance_reference",
        "effective_sample_size",
        "stability_contracted",
        "stability_contracted_provenance",
        "converged",
        "convergence_reason",
        "n_iter",
        "soft_non_convergence",
        # The canonical row-shape and column-shape path, entry by entry,
        # named so a reader can line a rolling entry up against the
        # static-reference entry of the same name. Built from the same
        # orientation: first index predicted, second predictor.
        *_SHAPE_ENTRY_COLUMNS,
    ),
    "rolling_increments.csv": (
        "period",
        "window",
        "d_row",
        "d_column",
        "d_gamma",
        "forecast_selected_pooling_rule",
    ),
    "regime_structure.csv": (
        "regime",
        "regime_label",
        "quantity",
        "row",
        "column",
        "value",
        "regime_interpretation_status",
    ),
    "transition_matrix.csv": (
        "from_regime",
        "to_regime",
        "probability",
        "expected_duration",
    ),
    "regime_neff.csv": (
        "model",
        "regime",
        "occupancy",
        "occupancy_share",
        "effective_sample_size",
        "innovation_variance",
        # Four conditions that prohibit economic interpretation, each
        # reported on its own with the threshold behind it.
        "condition_poor_occupancy",
        "condition_low_effective_sample",
        "condition_multistart_instability",
        "condition_gem_problem",
        "prohibits_economic_interpretation",
        "threshold_min_occupancy_share",
        "threshold_min_effective_sample",
        "threshold_multistart_loglik_tol",
        "threshold_multistart_min_agreeing_fraction",
        "threshold_provenance",
        "regime_interpretation_status",
    ),
    "regime_probabilities_smoothed.csv": (
        "period",
        "regime",
        "smoothed_probability",
        "smoothed_never_used_in_forecasts",
    ),
    "regime_probabilities_oos.csv": ("period", "regime", "predicted_probability"),
    "initialisation_diagnostics.csv": (
        "model",
        "fit_label",
        "init_id",
        "final_loglik",
        "retained",
        "converged",
        "convergence_reason",
    ),
    "estimator_telemetry.csv": (
        "model",
        "fit_label",
        "n_iter",
        "converged",
        "convergence_reason",
        "gem_rejected_steps",
        "rolled_back",
        "likelihood_monotone",
        "stability_contracted",
        "stability_contracted_provenance",
        "joint_spectral_radius",
        # Counts, not only the aggregate booleans beside them.
        "n_origins",
        "n_origins_contracted",
        "n_origins_soft_non_convergent",
        "convergence_assessed",
        "numerical_reliability_status",
    ),
    "raw_forecast_test.csv": (
        "model",
        "candidate",
        "rank",
        "window",
        "origin",
        "target",
        "fitting_preprocessor",
        "evaluation_preprocessor",
        "indicator",
        "country",
        "prediction_fit_scale",
        "prediction_eval_scale",
        "actual_eval_scale",
        "evaluation_error",
        "squared_error",
        "common_origin_eligible",
        "fit_status",
        "failure_reason",
    ),
    "rank_tuning_and_energy.csv": (
        "model",
        "rank",
        "canonical_parameter_count",
        "singular_value_energy_A",
        "singular_value_energy_B",
        "validation_msfe",
        "selected",
        "regularisation_not_true_rank",
    ),
}

# Required top-level keys of every registered JSON output.
JSON_SCHEMAS: dict[str, tuple] = {
    "empirical_design_manifest.json": (
        "panel",
        "samples",
        "counts",
        "provenance",
        "substitutions",
        "numerical_settings",
        "author_decisions",
        "simulation_benchmark",
        "origins",
        "failure_accounting",
    ),
    "preprocessing_full_sample.json": (
        "preprocessing_version",
        "information_cutoff",
        "seasonal_means",
        "scale_factors",
    ),
    "preprocessing_validation.json": (
        "preprocessing_version",
        "information_cutoff",
        "seasonal_means",
        "scale_factors",
    ),
    "preprocessing_test.json": (
        "preprocessing_version",
        "information_cutoff",
        "seasonal_means",
        "scale_factors",
    ),
}

# Tokens that must not appear in a registered output name or a column
# name.
PROHIBITED_TOKENS: tuple[str, ...] = (
    "sup_lm",
    "sup_wald",
    "suplm",
    "supwald",
    "break_date",
    "breakdate",
    "break_split",
    "trim",
    "bonferroni",
    "raw_score_block",
    "kronecker",
    "portmanteau",
    "ljung",
    "lag_order",
    "secondary_constancy",
    "pretest",
    "pre_test",
    "event_line",
    "shaded",
    "subperiod",
    "conventional_increment",
    "conv_d_b",
    "giacomini_rossi",
    "directional_accuracy",
    "probability_weighted",
)


def registered_files(mode: str) -> tuple[str, ...]:
    """Return every machine-readable file the mode must write.

    Parameters
    ----------
    mode : str
        Run mode.

    Returns
    -------
    tuple of str
        Sorted distinct file names.
    """
    names = {
        name
        for item in REGISTRY_ITEMS
        if mode in item.modes
        for name in item.files
    }
    return tuple(sorted(names))


def registered_figures(mode: str) -> tuple[str, ...]:
    """Return every figure the mode must write.

    Parameters
    ----------
    mode : str
        Run mode.

    Returns
    -------
    tuple of str
        Sorted distinct figure names.
    """
    names = {
        name
        for item in REGISTRY_ITEMS
        if mode in item.modes
        for name in item.figures
    }
    return tuple(sorted(names))


def items_for(mode: str) -> tuple[RegistryItem, ...]:
    """Return the registry items of one run mode.

    Parameters
    ----------
    mode : str
        Run mode.

    Returns
    -------
    tuple of RegistryItem
        The items.
    """
    return tuple(item for item in REGISTRY_ITEMS if mode in item.modes)


def summary_by_section(mode: str) -> dict:
    """Count registry entries per section.

    Parameters
    ----------
    mode : str
        Run mode.

    Returns
    -------
    dict
        Section prefix to item count.
    """
    counts: dict[str, int] = {}
    for item in items_for(mode):
        key = item.section.split(".")[0] + "." + item.section.split(".")[1] if "." in item.section else item.section
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
