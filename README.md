# mpmetrics 0.1.0

Librería de métricas robustas de tendencia central con ponderaciones adaptativas entre media, mediana, moda (opcional) y media recortada (opcional).


## Uso rápido
```python
import pandas as pd
from mpmetrics import metrica1, metrica2, metrica2_plot, diagnostic_plot

s = pd.Series([1,2,2,3,10])
res1 = metrica1(s)             # v1 (envuelve v2 con defaults)
res2 = metrica2(s)             # v2 (mejorada)
metrica2_plot(s)               # histograma + líneas
diagnostic_plot(s)             # KDE + marcas
```

## Accessor de pandas
```python
s.mp.metrica1()
s.mp.metrica1_plot()
s.mp.metrica2_plot()
s.mp.diagnostic_plot()
```

## Compatibilidad
- **Python**: 3.10 – 3.12  
- **NumPy**: ≥ 1.23  
- **Pandas**: ≥ 1.5  
- **SciPy**: ≥ 1.9  
- **Matplotlib**: ≥ 3.6


### Notas de entorno
- Esta librería **fuerza el backend `Agg`** de Matplotlib cuando no hay `MPLBACKEND` definido, para funcionar en entornos sin GUI (CI/servidores/pytest).  
  Si prefieres otro backend (ej. `TkAgg`), define:
  ```bash
  set MPLBACKEND=TkAgg   # Windows
  export MPLBACKEND=TkAgg  # Linux/Mac


### Troubleshooting

- **Error de `tk.tcl` / no hay GUI** al plotear durante `pytest` o en servidores:
  - La librería usa `Agg` automáticamente cuando `MPLBACKEND` no está definido.
  - Si quieres un backend interactivo, define `MPLBACKEND` a uno disponible (p. ej. `TkAgg`) y ten el runtime (Tk/Qt) instalado.

- **`NaN`/`Inf` en tu serie**:
  - Se limpian automáticamente. Si tras limpiar no quedan datos, se lanza `ValueError`.

- **Moda “no robusta”**:
  - Si fuerzas `incluir_moda=True` y el pico no cumple umbrales, emitimos `UserWarning`.  
    Puedes ajustar `min_peak_height` y `min_peak_width` si conoces tu distribución.