import pandas as pd
from mpmetrics import metrica1, metrica2

def test_basic_call():
    s = pd.Series([1,2,2,3,10])
    r1 = metrica1(s)
    r2 = metrica2(s)
    assert "tendencia_ponderada" in r1
    assert "tendencia_ponderada" in r2
