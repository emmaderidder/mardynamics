"""Reads the raw panel, checks it is complete, and prepares it for estimation.

Reads the raw indicator files, applies one approved substitution, checks
every cell is present, transforms each indicator, removes the seasonal
means of the active preprocessor, divides by its scale factors and
assembles the tensor.

Two preprocessing parameter sets exist, each estimated on data up to its
own cutoff date and held fixed after that point. The transformations are
one-sided differences.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import mardynamics.application.emp_config as emp_config


class DataIntegrityError(RuntimeError):
    """Raised when a country, indicator and month combination is missing
    from the panel.

    The exception message names the missing month, country and indicator.
    """


# =========================================================
# Reading the raw files
# =========================================================
def _file_hash(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_panel(data_dir: str | Path = "data") -> tuple[pd.DataFrame, list[dict]]:
    """Read the indicator files into one long panel.

    Reads one file per indicator in INDICATORS, keeps the countries in
    COUNTRIES, and records source metadata for each file alongside the
    values.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory holding the raw indicator files.

    Returns
    -------
    tuple
        The long panel with columns period, country, indicator and value,
        and a source-metadata record for each indicator.
    """
    directory = Path(data_dir)
    frames, provenance = [], []

    for indicator in emp_config.INDICATORS:
        path = directory / emp_config.INDICATOR_FILES[indicator]
        raw = pd.read_csv(path)
        keep = raw[raw["REF_AREA"].isin(emp_config.COUNTRIES)].copy()

        frames.append(
            pd.DataFrame(
                {
                    "period": keep["TIME_PERIOD"].astype(str),
                    "country": keep["REF_AREA"].astype(str),
                    "indicator": indicator,
                    "value": pd.to_numeric(keep["OBS_VALUE"], errors="coerce"),
                }
            )
        )

        provenance.append(
            {
                "indicator": indicator,
                "indicator_label": emp_config.INDICATOR_LABELS[indicator],
                "file": emp_config.INDICATOR_FILES[indicator],
                "file_sha256": _file_hash(path),
                "structure_id": _unique(keep, "STRUCTURE_ID"),
                "dataflow": _unique(keep, "STRUCTURE_NAME"),
                "measure": _unique(keep, "MEASURE"),
                "unit_measure": _unique(keep, "UNIT_MEASURE"),
                "adjustment": _unique(keep, "Adjustment"),
                "frequency": _unique(keep, "Frequency of observation"),
                "transformation_flag": _unique(keep, "Transformation"),
                "series_identifiers": sorted(
                    f"{indicator}:{country}" for country in emp_config.COUNTRIES
                ),
                "download_date": emp_config.DOWNLOAD_DATE,
                "n_raw_observations": int(len(keep)),
            }
        )

    panel = pd.concat(frames, ignore_index=True)
    return panel, provenance


def _unique(frame: pd.DataFrame, column: str) -> str:
    """Return the single distinct value of a metadata column.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw rows of one indicator.
    column : str
        Metadata column name.

    Returns
    -------
    str
        The distinct value, several values joined by "|", or an empty
        string when the column is absent.
    """
    if column not in frame:
        return ""
    values = sorted({str(v) for v in frame[column].dropna().unique()})
    return "|".join(values)


def apply_substitutions(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Insert the approved substitutions from APPROVED_SUBSTITUTIONS.

    The raw file itself is left untouched. A substituted value is added to
    the panel in memory and recorded with its source, value, retrieval
    date and reason.

    Parameters
    ----------
    panel : pandas.DataFrame
        Raw long panel.

    Returns
    -------
    tuple
        The panel including the substitutions, and a record of each one.
    """
    updated = panel.copy()
    applied = []

    for entry in emp_config.APPROVED_SUBSTITUTIONS:
        mask = (
            (updated["period"] == entry["period"])
            & (updated["country"] == entry["country"])
            & (updated["indicator"] == entry["indicator"])
        )
        present = bool(mask.any()) and bool(updated.loc[mask, "value"].notna().any())
        if present:
            applied.append({**entry, "applied": False, "note": "source value present"})
            continue

        updated = updated[~mask]
        updated = pd.concat(
            [
                updated,
                pd.DataFrame(
                    [
                        {
                            "period": entry["period"],
                            "country": entry["country"],
                            "indicator": entry["indicator"],
                            "value": float(entry["value"]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        applied.append({**entry, "applied": True, "note": "source value absent"})

    return updated, applied


def integrity_gate(panel: pd.DataFrame) -> None:
    """Check that every country/indicator/month cell is present.

    Parameters
    ----------
    panel : pandas.DataFrame
        Long panel, after substitution.

    Raises
    ------
    DataIntegrityError
        Names the country, indicator and month of every missing cell.
    """
    months = _month_index(emp_config.RAW_START, emp_config.RAW_END)
    present = {
        (row.period, row.country, row.indicator)
        for row in panel[panel["value"].notna()].itertuples()
    }

    missing = [
        {"period": month, "country": country, "indicator": indicator}
        for month in months
        for country in emp_config.COUNTRIES
        for indicator in emp_config.INDICATORS
        if (month, country, indicator) not in present
    ]

    if missing:
        detail = "; ".join(
            f"{item['country']}/{item['indicator']}/{item['period']}"
            for item in missing[:20]
        )
        raise DataIntegrityError(
            f"{len(missing)} missing panel cells, no interpolation permitted: {detail}"
        )


def _month_index(start: str, end: str) -> list[str]:
    """Return the inclusive list of months between two YYYY-MM bounds.

    Parameters
    ----------
    start, end : str
        First and last month, as YYYY-MM.

    Returns
    -------
    list of str
        The months.
    """
    return [str(p) for p in pd.period_range(start, end, freq="M")]


# =========================================================
# Transformations
# =========================================================
def transform_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the stationarity transformation of each indicator.

    Indicators in LOG_DIFF_INDICATORS enter as first log differences within
    each country; the remaining indicators enter as first differences in
    levels. Both are one-sided in time.

    Parameters
    ----------
    panel : pandas.DataFrame
        Raw long panel.

    Returns
    -------
    pandas.DataFrame
        The transformed panel, over the transformed sample window.
    """
    wide = panel.pivot_table(
        index="period", columns=["indicator", "country"], values="value"
    ).sort_index()

    pieces = []
    for indicator in emp_config.INDICATORS:
        block = wide[indicator][list(emp_config.COUNTRIES)]
        if indicator in emp_config.LOG_DIFF_INDICATORS:
            transformed = np.log(block).diff()
        else:
            transformed = block.diff()
        transformed.columns = pd.MultiIndex.from_product(
            [[indicator], list(emp_config.COUNTRIES)]
        )
        pieces.append(transformed)

    result = pd.concat(pieces, axis=1)
    months = _month_index(emp_config.SAMPLE_START, emp_config.SAMPLE_END)
    return result.loc[months]


# =========================================================
# The preprocessors
# =========================================================
@dataclass
class Preprocessor:
    """One preprocessing parameter set: seasonal means and scale factors.

    Attributes
    ----------
    version : str
        Label of this parameter set.
    cutoff : str
        Last month whose data the parameters may use.
    seasonal_means : dict
        Seasonal mean for each indicator in SEASONAL_INDICATORS, keyed by
        "<country>-<month>".
    scales : dict
        Pooled scale factor for each indicator.
    n_estimation_months : int
        Number of months the parameters were estimated on.
    """

    version: str
    cutoff: str
    seasonal_means: dict
    scales: dict
    n_estimation_months: int

    def to_payload(self) -> dict:
        """Return the parameters as a JSON-serialisable dict.

        Returns
        -------
        dict
            The parameters and the window they were estimated on.
        """
        return {
            "preprocessing_version": self.version,
            "information_cutoff": self.cutoff,
            "estimation_window": [emp_config.SAMPLE_START, self.cutoff],
            "n_estimation_months": self.n_estimation_months,
            "n_seasonal_means": len(self.seasonal_means),
            "n_scale_factors": len(self.scales),
            "seasonal_means": self.seasonal_means,
            "scale_factors": self.scales,
            "seasonally_adjusted_indicators": list(emp_config.SEASONAL_INDICATORS),
            "log_difference_indicators": list(emp_config.LOG_DIFF_INDICATORS),
            "level_difference_indicators": list(emp_config.DIFF_INDICATORS),
        }


def estimate_preprocessor(transformed: pd.DataFrame, version: str) -> Preprocessor:
    """Estimate the seasonal means and scale factors for one preprocessing
    version.

    The seasonal means are country-specific calendar-month averages of the
    indicators in SEASONAL_INDICATORS, computed over the window ending at
    the version's cutoff date. The scale factors are one pooled standard
    deviation per indicator, computed after removing the seasonal means.

    Parameters
    ----------
    transformed : pandas.DataFrame
        Transformed panel, before standardisation.
    version : {"validation", "test"}
        Which preprocessing version to estimate.

    Returns
    -------
    Preprocessor
        The estimated seasonal means and scale factors.
    """
    cutoff = emp_config.PREPROCESSOR_CUTOFFS[version]
    window = transformed.loc[transformed.index <= cutoff]

    seasonal_means = {}
    months = pd.PeriodIndex(window.index, freq="M").month
    for indicator in emp_config.SEASONAL_INDICATORS:
        for country in emp_config.COUNTRIES:
            series = window[(indicator, country)]
            for month in range(1, 13):
                selection = series[months == month]
                seasonal_means[f"{country}-{month:02d}"] = float(
                    selection.mean(skipna=True)
                )

    adjusted = _subtract_seasonal_means(window, seasonal_means)
    scales = {
        indicator: float(
            np.std(
                adjusted[indicator][list(emp_config.COUNTRIES)].to_numpy(dtype=float).ravel(),
                ddof=1,
            )
        )
        for indicator in emp_config.INDICATORS
    }

    return Preprocessor(
        version=version,
        cutoff=cutoff,
        seasonal_means=seasonal_means,
        scales=scales,
        n_estimation_months=int(len(window)),
    )


def _subtract_seasonal_means(frame: pd.DataFrame, seasonal_means: dict) -> pd.DataFrame:
    """Subtract the estimated seasonal means from the seasonal indicators.

    Parameters
    ----------
    frame : pandas.DataFrame
        A transformed panel.
    seasonal_means : dict
        Seasonal means, as returned by estimate_preprocessor.

    Returns
    -------
    pandas.DataFrame
        The panel with the seasonal indicators seasonally adjusted.
    """
    adjusted = frame.copy()
    months = pd.PeriodIndex(frame.index, freq="M").month
    for indicator in emp_config.SEASONAL_INDICATORS:
        for country in emp_config.COUNTRIES:
            offsets = np.array(
                [seasonal_means[f"{country}-{month:02d}"] for month in months]
            )
            adjusted[(indicator, country)] = (
                frame[(indicator, country)].to_numpy(dtype=float) - offsets
            )
    return adjusted


def map_scale(
    matrix: np.ndarray, source: Preprocessor, target: Preprocessor, period: str
) -> np.ndarray:
    """Convert a standardised matrix from one preprocessing scale to another.

    For indicator i, country c and calendar month m:

    x_target = (x_source * scale_source[i] + a_source[c, m] - a_target[c, m])
    / scale_target[i]

    where the seasonal terms are zero for indicators outside
    SEASONAL_INDICATORS. Apply this to a forecast or an actual, not to an
    estimated model parameter.

    Parameters
    ----------
    matrix : numpy.ndarray
        A (m, n) matrix standardised on the source scale.
    source, target : Preprocessor
        Parameter sets of the source and target scales.
    period : str
        Calendar month of the matrix, as YYYY-MM.

    Returns
    -------
    numpy.ndarray
        The matrix restated on the target scale.
    """
    if source.version == target.version:
        return np.array(matrix, dtype=float, copy=True)

    month = int(period[-2:])
    mapped = np.empty_like(matrix, dtype=float)
    for i, indicator in enumerate(emp_config.INDICATORS):
        for j, country in enumerate(emp_config.COUNTRIES):
            value = float(matrix[i, j]) * source.scales[indicator]
            if indicator in emp_config.SEASONAL_INDICATORS:
                key = f"{country}-{month:02d}"
                value += source.seasonal_means[key] - target.seasonal_means[key]
            mapped[i, j] = value / target.scales[indicator]
    return mapped


def apply_preprocessor(
    transformed: pd.DataFrame, preprocessor: Preprocessor
) -> pd.DataFrame:
    """Standardise the transformed panel with one preprocessing parameter set.

    Parameters
    ----------
    transformed : pandas.DataFrame
        Transformed panel.
    preprocessor : Preprocessor
        Parameter set to apply.

    Returns
    -------
    pandas.DataFrame
        The standardised panel.
    """
    adjusted = _subtract_seasonal_means(transformed, preprocessor.seasonal_means)
    standardised = adjusted.copy()
    for indicator in emp_config.INDICATORS:
        scale = preprocessor.scales[indicator]
        for country in emp_config.COUNTRIES:
            standardised[(indicator, country)] = (
                adjusted[(indicator, country)].to_numpy(dtype=float) / scale
            )
    return standardised


# =========================================================
# Tensor assembly
# =========================================================
def build_tensor(standardised: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Assemble the T x m x n tensor, with indicators on rows.

    Parameters
    ----------
    standardised : pandas.DataFrame
        Standardised panel.

    Returns
    -------
    tuple
        The tensor and its ordered list of periods.

    Raises
    ------
    DataIntegrityError
        If the assembled tensor contains a non-finite entry.
    """
    periods = [str(p) for p in standardised.index]
    tensor = np.empty((len(periods), emp_config.M_DIM, emp_config.N_DIM), dtype=float)
    for i, indicator in enumerate(emp_config.INDICATORS):
        for j, country in enumerate(emp_config.COUNTRIES):
            tensor[:, i, j] = standardised[(indicator, country)].to_numpy(dtype=float)

    if not np.isfinite(tensor).all():
        raise DataIntegrityError("the assembled tensor contains non-finite entries")
    return tensor, periods


# =========================================================
# ADF diagnostics
# =========================================================
def adf_diagnostics(standardised: pd.DataFrame) -> pd.DataFrame:
    """Run an augmented Dickey-Fuller test on every indicator/country series.

    Uses a constant and no trend, the Schwert maximum lag
    floor(12 (T/100)^(1/4)), AIC lag selection and MacKinnon p-values.

    Parameters
    ----------
    standardised : pandas.DataFrame
        Transformed and standardised full-sample panel.

    Returns
    -------
    pandas.DataFrame
        One row per indicator and country.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for indicator in emp_config.INDICATORS:
        for country in emp_config.COUNTRIES:
            series = standardised[(indicator, country)].to_numpy(dtype=float)
            statistic, pvalue, used_lag, n_obs, critical, _ = adfuller(
                series,
                maxlag=emp_config.ADF_MAX_LAG,
                regression=emp_config.ADF_REGRESSION,
                autolag=emp_config.ADF_AUTOLAG,
            )
            rows.append(
                {
                    "indicator": indicator,
                    "indicator_label": emp_config.INDICATOR_LABELS[indicator],
                    "country": country,
                    "country_label": emp_config.COUNTRY_LABELS[country],
                    "adf_statistic": float(statistic),
                    "p_value": float(pvalue),
                    "selected_lag": int(used_lag),
                    "max_lag": int(emp_config.ADF_MAX_LAG),
                    "effective_observations": int(n_obs),
                    "regression": emp_config.ADF_REGRESSION,
                    "lag_selection": emp_config.ADF_AUTOLAG,
                    "critical_1pct": float(critical["1%"]),
                    "critical_5pct": float(critical["5%"]),
                    "critical_10pct": float(critical["10%"]),
                }
            )
    return pd.DataFrame(rows)


# =========================================================
# The complete data preparation route
# =========================================================
@dataclass
class EmpiricalData:
    """The prepared panel and the objects derived from it.

    Attributes
    ----------
    provenance : list of dict
        Source metadata of the raw files.
    substitutions : list of dict
        Applied substitutions.
    transformed : pandas.DataFrame
        Transformed panel, before standardisation.
    preprocessors : dict
        The preprocessing parameter sets, by version.
    tensors : dict
        Standardised tensor for each preprocessing version.
    periods : list of str
        Ordered months of the transformed sample.
    adf : pandas.DataFrame
        ADF diagnostics on the full-sample panel.
    """

    provenance: list
    substitutions: list
    transformed: pd.DataFrame
    preprocessors: dict
    tensors: dict
    periods: list
    adf: pd.DataFrame


def prepare(data_dir: str | Path = "data") -> EmpiricalData:
    """Read, check, transform and standardise the panel.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory holding the raw indicator files.

    Returns
    -------
    EmpiricalData
        The prepared data.

    Raises
    ------
    DataIntegrityError
        When a cell is missing or an expected count is not met.
    """
    panel, provenance = load_raw_panel(data_dir)
    panel, substitutions = apply_substitutions(panel)
    integrity_gate(panel)

    transformed = transform_panel(panel)
    if len(transformed) != emp_config.EXPECTED_MATRICES:
        raise DataIntegrityError(
            f"expected {emp_config.EXPECTED_MATRICES} matrices, found {len(transformed)}"
        )
    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise DataIntegrityError("the transformed panel contains non-finite entries")

    preprocessors, tensors, periods = {}, {}, None
    for version in emp_config.PREPROCESSOR_CUTOFFS:
        preprocessor = estimate_preprocessor(transformed, version)
        standardised = apply_preprocessor(transformed, preprocessor)
        tensor, periods = build_tensor(standardised)
        preprocessors[version] = preprocessor
        tensors[version] = tensor

    for version, tensor in tensors.items():
        if tensor.shape[0] - 1 != emp_config.EXPECTED_TRANSITIONS:
            raise DataIntegrityError(
                f"expected {emp_config.EXPECTED_TRANSITIONS} transitions on the "
                f"{version} preprocessor, found {tensor.shape[0] - 1}"
            )

    # The full-sample preprocessor is used for the retrospective diagnostics.
    adf = adf_diagnostics(
        apply_preprocessor(
            transformed, preprocessors[emp_config.SECTION_PREPROCESSING["5.1_adf"]]
        )
    )

    return EmpiricalData(
        provenance=provenance,
        substitutions=substitutions,
        transformed=transformed,
        preprocessors=preprocessors,
        tensors=tensors,
        periods=periods,
        adf=adf,
    )


# =========================================================
# Forecast origins
# =========================================================
def forecast_origins(periods: list[str], phase: str, mode: str) -> list[str]:
    """Return the ordered forecast target months of one phase.

    Parameters
    ----------
    periods : list of str
        Ordered months of the transformed sample.
    phase : {"validation", "test"}
        Which forecast phase.
    mode : {"preview", "full"}
        Run mode. In preview mode, every fourth ordered origin is used,
        starting from the first.

    Returns
    -------
    list of str
        The target months to forecast.
    """
    start, end = (
        (emp_config.VALIDATION_START, emp_config.VALIDATION_END)
        if phase == "validation"
        else (emp_config.TEST_START, emp_config.TEST_END)
    )
    ordered = [p for p in periods if start <= p <= end]
    if mode == "preview":
        return ordered[:: emp_config.PREVIEW_ORIGIN_STRIDE]
    return ordered
