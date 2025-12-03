from __future__ import annotations
from typing import Optional, Iterable
import warnings
import os
import matplotlib as mpl
# Usa backend no interactivo por defecto para entornos sin GUI (pytest/CI).
# Si el usuario define MPLBACKEND, respetamos su elección.
if os.environ.get("MPLBACKEND") is None:
    try:
        mpl.use("Agg")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .core import metrica_ponderada_v2

def _prepare_input(x: Iterable, name: str = "x") -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} debe ser 1D (una sola columna). Recibí {x.shape[1]} columnas.")
        arr = x.iloc[:, 0].to_numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} debe ser 1D. Recibí {arr.ndim}D.")
    arr = arr.astype(float, copy=False)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} no contiene valores numéricos finitos tras limpiar NaN/Inf.")
    return arr

def metrica1(x: Iterable, **params) -> pd.Series:
    arr = _prepare_input(x)
    defaults = dict(incluir_trimmed=False, use_kurtosis=False, use_kurtosis_bell=False)
    defaults.update(params)
    res = metrica_ponderada_v2(arr, **defaults)
    if defaults.get("incluir_moda") and not bool(res.get("moda_robusta", False)):
        warnings.warn(
            "Se solicitó incluir la moda, pero el pico no fue robusto según los umbrales establecidos.",
            UserWarning,
            stacklevel=2,
        )
    return res

def metrica2(x: Iterable, **params) -> pd.Series:
    arr = _prepare_input(x)
    res = metrica_ponderada_v2(arr, **params)
    if params.get("incluir_moda") and not bool(res.get("moda_robusta", False)):
        warnings.warn(
            "Se solicitó incluir la moda, pero el pico no fue robusto según los umbrales establecidos.",
            UserWarning,
            stacklevel=2,
        )
    return res

def metrica1_plot(x: Iterable, result: Optional[pd.Series] = None, bins: int = 30, ax=None, show: bool = True, **params):
    arr = _prepare_input(x)
    if result is None:
        result = metrica1(arr, **params)
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(arr, bins=bins, alpha=0.3, density=True)
    ax.axvline(result.get("media"), linestyle="--", label="media")
    ax.axvline(result.get("mediana"), linestyle=":", label="mediana")
    ax.axvline(result.get("tendencia_ponderada"), linestyle="-", label="tendencia_ponderada")
    ax.legend()
    ax.set_title("metrica1_plot")
    if show:
        plt.show()
    return ax

def metrica2_plot(x: Iterable, result: Optional[pd.Series] = None, bins: int = 30, ax=None, show: bool = True, **params):
    arr = _prepare_input(x)
    if result is None:
        result = metrica2(arr, **params)
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(arr, bins=bins, alpha=0.3, density=True)
    ax.axvline(result.get("media"), linestyle="--", label="media")
    ax.axvline(result.get("mediana"), linestyle=":", label="mediana")
    if np.isfinite(result.get("moda", np.nan)):
        ax.axvline(result.get("moda"), linestyle="-.", label="moda")
    ax.axvline(result.get("tendencia_ponderada"), linestyle="-", label="tendencia_ponderada")
    ax.legend()
    ax.set_title("metrica2_plot")
    if show:
        plt.show()
    return ax

def diagnostic_plot(
    x: Iterable,
    result: Optional[pd.Series] = None,
    kde_points: int = 512,
    ax=None,
    show: bool = True,
    **params
):
    arr = _prepare_input(x)
    if result is None:
        result = metrica2(arr, **params)
    if ax is None:
        fig, ax = plt.subplots()
    kde = gaussian_kde(arr)
    grid = np.linspace(arr.min(), arr.max(), kde_points)
    dens = kde(grid)
    ax.plot(grid, dens)
    if np.isfinite(result.get("media", np.nan)):
        ax.axvline(result.get("media"), linestyle="--", label="media")
    if np.isfinite(result.get("mediana", np.nan)):
        ax.axvline(result.get("mediana"), linestyle=":", label="mediana")
    if np.isfinite(result.get("moda", np.nan)):
        ax.axvline(result.get("moda"), linestyle="-.", label="moda")
    if np.isfinite(result.get("tendencia_ponderada", np.nan)):
        ax.axvline(result.get("tendencia_ponderada"), linestyle="-", label="tendencia_ponderada")
    ax.legend()
    ax.set_title("diagnostic_plot (KDE + referencias)")
    if show:
        plt.show()
    return ax
