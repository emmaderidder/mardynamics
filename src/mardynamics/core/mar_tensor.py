"""Tensor-structure utilities for the matrix autoregressive (MAR) models.

This module provides the helpers that slice the ``T x m x n`` MAR tensor into
one-step lagged predictor/response pairs and map forecast matrices back to
flat ``variable_country`` column labels. It contains pure data-reshaping logic
only; the actual estimation lives in ``mar_fit`` and ``regime_switching_em``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_mar_column_names(variables: list[str], countries: list[str]) -> list[str]:
    """Build flat ``variable_country`` column names in MAR forecast order.

    The ordering iterates variables on the outer loop and countries on the inner
    loop, matching the row-major (C-order) flattening of an ``m x n`` forecast
    matrix.

    Parameters
    ----------
    variables : list of str
        Variable names (the row / ``m`` dimension).
    countries : list of str
        Country names (the column / ``n`` dimension).

    Returns
    -------
    list of str
        Column labels of the form ``"{variable}_{country}"``.
    """
    flat_col_names = [f"{var}_{country}" for var in variables for country in countries]
    return flat_col_names


def split_mar_lagged_tensor(train_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a tensor into one-step lagged predictor/response pairs.

    For observations ``X_0, ..., X_{T-1}`` this returns ``X_prev[t] = X_t`` and
    ``Y[t] = X_{t+1}``; each response is the one-step-ahead observation of its
    predictor.

    Parameters
    ----------
    train_tensor : numpy.ndarray
        Array of shape ``(T, m, n)`` with at least two time points.

    Returns
    -------
    X_prev : numpy.ndarray
        Predictors ``X_0, ..., X_{T-2}`` (shape ``(T-1, m, n)``).
    Y : numpy.ndarray
        Responses ``X_1, ..., X_{T-1}`` (shape ``(T-1, m, n)``).

    Raises
    ------
    ValueError
        If ``train_tensor`` is not 3-D or has fewer than two time points.
    """
    train_tensor = np.asarray(train_tensor, dtype=float)

    if train_tensor.ndim != 3:
        raise ValueError("train_tensor must be a 3D array with dimensions T x m x n.")

    if train_tensor.shape[0] < 2:
        raise ValueError("train_tensor must contain at least 2 time points.")

    X_prev = train_tensor[:-1]
    Y = train_tensor[1:]

    return X_prev, Y


def get_last_mar_tensor_observation(train_tensor: np.ndarray) -> np.ndarray:
    """Return a copy of the final observation matrix in the tensor.

    Parameters
    ----------
    train_tensor : numpy.ndarray
        Array of shape ``(T, m, n)`` with at least one time point.

    Returns
    -------
    numpy.ndarray
        Copy of the last ``m x n`` slice, used as the forecast origin ``X_T``.

    Raises
    ------
    ValueError
        If ``train_tensor`` is not 3-D or is empty.
    """
    train_tensor = np.asarray(train_tensor, dtype=float)

    if train_tensor.ndim != 3:
        raise ValueError("train_tensor must be a 3D array with dimensions T x m x n.")

    if train_tensor.shape[0] < 1:
        raise ValueError("train_tensor must contain at least 1 time point.")

    return train_tensor[-1].copy()


def flatten_mar_forecast_matrix(
    forecast_matrix: np.ndarray,
    variables: list[str],
    countries: list[str],
) -> pd.Series:
    """Flatten an ``m x n`` forecast matrix into a labelled MAR-order series.

    Parameters
    ----------
    forecast_matrix : numpy.ndarray
        Forecast matrix of shape ``(len(variables), len(countries))``.
    variables : list of str
        Variable names (row / ``m`` dimension).
    countries : list of str
        Country names (column / ``n`` dimension).

    Returns
    -------
    pandas.Series
        The flattened forecast in row-major (C) order, indexed by the
        ``variable_country`` labels.

    Raises
    ------
    ValueError
        If ``forecast_matrix`` does not have shape
        ``(len(variables), len(countries))``.
    """
    forecast_matrix = np.asarray(forecast_matrix, dtype=float)

    expected_shape = (len(variables), len(countries))
    if forecast_matrix.shape != expected_shape:
        raise ValueError(
            f"forecast_matrix must have shape {expected_shape}, "
            f"got {forecast_matrix.shape}."
        )

    flat_col_names = _build_mar_column_names(variables, countries)
    flat_forecast = forecast_matrix.reshape(-1, order="C")

    return pd.Series(flat_forecast, index=flat_col_names)
