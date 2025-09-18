import numpy as np
import pandas as pd
import pytest
from mpmetrics import metrica2, metrica2_plot, diagnostic_plot

def test_metrica2_plot_returns_axes():
    s = pd.Series(np.random.default_rng(0).normal(size=200))
    ax = metrica2_plot(s, show=False)
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)

def test_diagnostic_plot_returns_axes():
    s = pd.Series(np.random.default_rng(1).normal(size=300))
    ax = diagnostic_plot(s, show=False)
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)

def test_warning_when_moda_not_robust():
    s = pd.Series(np.random.default_rng(2).normal(size=300))
    # Forzamos umbrales imposibles para que la función marque "no robusta" y emita warning
    with pytest.warns(UserWarning):
        metrica2(s, incluir_moda=True, moda_robusta=True, min_peak_height=10.0, min_peak_width=1.0)
