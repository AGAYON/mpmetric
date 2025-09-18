import numpy as np
import pandas as pd
from mpmetrics import metrica1, metrica2

def test_nan_inf_cleaning():
    s = pd.Series([1, 2, np.nan, np.inf, 3, -np.inf, 4])
    r = metrica1(s)
    assert np.isfinite(r["tendencia_ponderada"])

def test_small_sample():
    s = pd.Series([10.0, 12.0])
    r = metrica2(s)
    assert "tendencia_ponderada" in r and np.isfinite(r["tendencia_ponderada"])

def test_mesocurtic_prefers_mean():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=5000)
    r = metrica2(x)
    assert r["peso_media"] >= 0.5

def test_trimmed_included_weights():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 1000), rng.normal(0, 5, 200)])
    r = metrica2(x, incluir_trimmed=True)
    assert r["peso_trimmed"] > 0
