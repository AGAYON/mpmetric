import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

def madn(x):
    return 1.4826 * mad(x)

def bowley_skew(x):
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return (q3 + q1 - 2*q2) / iqr

def excess_kurtosis(x):
    x = np.asarray(x)
    n = len(x)
    if n < 4:
        return 0.0
    m = x.mean()
    s2 = np.mean((x - m)**2)
    if s2 == 0:
        return 0.0
    m4 = np.mean((x - m)**4)
    g2 = m4 / (s2**2) - 3.0
    return g2

def trimmed_mean(x, prop=0.1):
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.floor(prop * n))
    if n - 2*k <= 0:
        return float(np.mean(x))
    return float(np.mean(x[k:n-k]))

def moda_kde(x, bw_method='scott'):
    x = np.asarray(x)
    kde = gaussian_kde(x, bw_method=bw_method)
    grid = np.linspace(np.min(x), np.max(x), 1000)
    dens = kde(grid)
    return float(grid[np.argmax(dens)])

def moda_kde_robusta(x, bw_method='scott', grid_size=2000, usar_expansion=False,
                     columna_expansion=None, expansion_data=None, expansion_factor=1.2,
                     min_peak_height=0.1, min_peak_width=0.05):
    x = np.asarray(x)
    if len(x) < 3:
        return {"moda": np.median(x), "altura_relativa": 0.0, "ancho_pico": 0.0, "es_robusta": False}
    if usar_expansion and columna_expansion is not None and expansion_data is not None:
        if hasattr(expansion_data, columna_expansion):
            factores = expansion_data[columna_expansion].values
        elif isinstance(expansion_data, dict) and columna_expansion in expansion_data:
            factores = np.array(expansion_data[columna_expansion])
        else:
            factores = np.ones(len(x))
        if len(factores) != len(x):
            factores = np.ones(len(x))
        kde = gaussian_kde(x, bw_method=bw_method, weights=factores/np.sum(factores))
        expansion_range = np.mean(factores)
    else:
        kde = gaussian_kde(x, bw_method=bw_method)
        expansion_range = expansion_factor
    x_min, x_max = np.min(x), np.max(x)
    rango = x_max - x_min
    grid_min = x_min - (expansion_range - 1) * rango / 2
    grid_max = x_max + (expansion_range - 1) * rango / 2
    grid = np.linspace(grid_min, grid_max, grid_size)
    dens = kde(grid)
    max_idx = np.argmax(dens); max_density = dens[max_idx]; moda_value = grid[max_idx]
    altura_relativa = max_density / np.mean(dens) if np.mean(dens) > 0 else 0.0
    half_max = max_density / 2
    left_idx = max_idx; right_idx = max_idx
    while left_idx > 0 and dens[left_idx] > half_max: left_idx -= 1
    while right_idx < len(dens) - 1 and dens[right_idx] > half_max: right_idx += 1
    ancho_pico = (grid[right_idx] - grid[left_idx]) / rango if rango > 0 else 0.0
    es_robusta = (altura_relativa >= min_peak_height and ancho_pico >= min_peak_width and grid_min <= moda_value <= grid_max)
    return {"moda": float(moda_value), "altura_relativa": float(altura_relativa), "ancho_pico": float(ancho_pico),
            "es_robusta": bool(es_robusta), "densidad_maxima": float(max_density)}

def weight_exponential(s, alpha=0.693):
    w_mean = np.exp(-alpha * s); return float(np.clip(w_mean, 0.0, 1.0))

def weight_logistic(s, s0=1.0, p=2.0):
    w_mean = 1.0 / (1.0 + (s / max(s0, 1e-12))**p); return float(np.clip(w_mean, 0.0, 1.0))

def weight_linear(s, s_max=2.0):
    w_mean = max(0.0, 1.0 - s / max(s_max, 1e-12)); return float(np.clip(w_mean, 0.0, 1.0))

def adjust_by_kurtosis(w_mean, g2, beta=0.25):
    factor = np.exp(-beta * max(0.0, g2)); return float(np.clip(w_mean * factor, 0.0, 1.0))

def adjust_by_kurtosis_bell(w_mean, g2, reward_window=0.5, reward_gain=0.08,
                            beta_neg=0.15, beta_pos=0.25):
    g2_abs = abs(g2)
    if g2_abs <= reward_window:
        factor = 1.0 + reward_gain * (1.0 - g2_abs / max(reward_window, 1e-12))
    else:
        over = g2_abs - reward_window
        beta = beta_pos if g2 >= 0 else beta_neg
        factor = np.exp(-beta * over)
    return float(np.clip(w_mean * factor, 0.0, 1.0))

def shrink_s_by_n(s, n, c=100.0):
    return float(s * np.sqrt(n / (n + c)))

def softmax(vals, temperature=1.0):
    vals = np.array(vals) / max(temperature, 1e-12)
    exp_vals = np.exp(vals - np.max(vals))
    return exp_vals / np.sum(exp_vals)

def convex_weights(distances, method='inverse_distance', alpha=2.0):
    distances = np.array(distances) + 1e-12
    if method == 'inverse_distance':
        weights = 1 / (distances ** alpha)
    elif method == 'exponential':
        weights = np.exp(-alpha * distances)
    elif method == 'polynomial':
        weights = 1 / (1 + distances ** alpha)
    else:
        raise ValueError("method debe ser 'inverse_distance', 'exponential' o 'polynomial'")
    return weights / np.sum(weights)

def metrica_ponderada_v2(
    x,
    method="logistic",
    usar_medida_robusta=True,
    usar_transformacion_no_lineal=True,
    ajustar_por_n=True,
    use_kurtosis=False,
    use_kurtosis_bell=False,
    use_bowley=False,
    incluir_moda=False,
    moda_robusta=False,
    incluir_trimmed=False,
    trimmed_prop=0.1,
    weight_method='softmax',
    convex_method='inverse_distance',
    temperature=0.5,
    alpha=0.693,
    s0=1.0,
    p=2.0,
    s_max=2.0,
    shrink_c=100.0,
    clip=(0.05, 0.95),
    bw_method='scott',
    grid_size=2000,
    usar_expansion=False,
    columna_expansion=None,
    expansion_data=None,
    expansion_factor=1.2,
    min_peak_height=0.1,
    min_peak_width=0.05,
    w_media_floor=None
):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n == 0:
        return pd.Series({
            "n": 0, "media": np.nan, "mediana": np.nan, "moda": np.nan, "trimmed": np.nan,
            "MADN": np.nan, "bowley": np.nan, "exceso_kurtosis": np.nan,
            "peso_media": np.nan, "peso_mediana": np.nan, "peso_moda": np.nan, "peso_trimmed": np.nan,
            "tendencia_ponderada": np.nan, "moda_robusta": False, "altura_pico": np.nan, "ancho_pico": np.nan
        })

    media = float(np.mean(x))
    mediana = float(np.median(x))
    trimmed = float(trimmed_mean(x, trimmed_prop)) if incluir_trimmed else np.nan

    if incluir_moda:
        if moda_robusta:
            moda_info = moda_kde_robusta(
                x, bw_method=bw_method, grid_size=grid_size,
                usar_expansion=usar_expansion, columna_expansion=columna_expansion,
                expansion_data=expansion_data, expansion_factor=expansion_factor,
                min_peak_height=min_peak_height, min_peak_width=min_peak_width
            )
            moda = moda_info["moda"]
            es_moda_robusta = moda_info["es_robusta"]
            altura_pico = moda_info["altura_relativa"]
            ancho_pico = moda_info["ancho_pico"]
        else:
            moda = float(moda_kde(x, bw_method=bw_method))
            es_moda_robusta = True
            altura_pico = 1.0
            ancho_pico = 1.0
    else:
        moda = np.nan
        es_moda_robusta = False
        altura_pico = np.nan
        ancho_pico = np.nan

    escala = madn(x) if usar_medida_robusta else np.std(x)
    if escala == 0:
        escala = 1e-12

    s_mean_med = abs(media - mediana) / escala
    s_mode_med = 0.0 if not incluir_moda else abs(mediana - moda) / escala
    s_mean_mode = 0.0 if not incluir_moda else abs(media - moda) / escala
    s_trim_med = 0.0 if not incluir_trimmed else abs(trimmed - mediana) / escala
    s_mean_trim = 0.0 if not incluir_trimmed else abs(media - trimmed) / escala

    bowley_asimetria = bowley_skew(x)
    g2 = excess_kurtosis(x) if use_kurtosis or use_kurtosis_bell else 0.0

    if ajustar_por_n:
        s_mean_med = shrink_s_by_n(s_mean_med, n, c=shrink_c)
        if incluir_moda:
            s_mode_med = shrink_s_by_n(s_mode_med, n, c=shrink_c)
            s_mean_mode = shrink_s_by_n(s_mean_mode, n, c=shrink_c)
        if incluir_trimmed:
            s_trim_med = shrink_s_by_n(s_trim_med, n, c=shrink_c)
            s_mean_trim = shrink_s_by_n(s_mean_trim, n, c=shrink_c)

    if incluir_moda and moda_robusta and not es_moda_robusta:
        s_mode_med *= 2.0
        s_mean_mode *= 2.0

    centros = []
    distancias = []

    if incluir_moda and incluir_trimmed:
        centros = ['media', 'mediana', 'moda', 'trimmed']
        distancias = np.array([s_mean_med, s_mode_med, s_mean_mode + 1e-12, s_trim_med])
        scores = -distancias
    elif incluir_moda:
        centros = ['media', 'mediana', 'moda']
        distancias = np.array([s_mean_med, s_mode_med, s_mean_mode + 1e-12])
        scores = -distancias
    elif incluir_trimmed:
        centros = ['media', 'mediana', 'trimmed']
        distancias = np.array([s_mean_med, s_trim_med, s_mean_trim + 1e-12])
        scores = -distancias
    else:
        centros = ['media', 'mediana']
        distancias = np.array([s_mean_med, 1e-12])
        scores = -distancias

    if usar_transformacion_no_lineal:
        if method == "exponential":
            w_media = weight_exponential(s_mean_med, alpha=alpha)
        elif method == "logistic":
            w_media = weight_logistic(s_mean_med, s0=s0, p=p)
        elif method == "linear":
            w_media = weight_linear(s_mean_med, s_max=s_max)
        else:
            raise ValueError("method debe ser 'exponential', 'logistic' o 'linear'")
    else:
        w_media = max(0.0, 1.0 - s_mean_med)

    if use_kurtosis:
        w_media = adjust_by_kurtosis(w_media, g2, beta=0.25)
    if use_kurtosis_bell:
        w_media = adjust_by_kurtosis_bell(w_media, g2, reward_window=0.5, reward_gain=0.08,
                                          beta_neg=0.15, beta_pos=0.25)
    if use_bowley:
        bowley_factor = np.exp(-0.2 * abs(bowley_asimetria))
        w_media *= bowley_factor

    if w_media_floor is not None:
        w_media = max(w_media, float(w_media_floor))

    lo, hi = clip
    w_media = float(np.clip(w_media, lo, hi))

    if len(centros) == 2:
        peso_media = w_media
        peso_mediana = 1.0 - w_media
        peso_moda = 0.0
        peso_trim = 0.0
        tendencia = peso_media * media + peso_mediana * mediana
    else:
        if weight_method == 'softmax':
            pesos = softmax(scores, temperature=temperature)
        elif weight_method == 'convex':
            pesos = convex_weights(distancias, method=convex_method, alpha=temperature)
        else:
            raise ValueError("weight_method debe ser 'softmax' o 'convex'")
        mapa = dict(zip(centros, pesos))
        peso_media = float(mapa.get('media', 0.0))
        peso_mediana = float(mapa.get('mediana', 0.0))
        peso_moda = float(mapa.get('moda', 0.0))
        peso_trim = float(mapa.get('trimmed', 0.0))
        total = peso_media + peso_mediana + peso_moda + peso_trim
        if total <= 0:
            peso_media, peso_mediana = 0.5, 0.5
            peso_moda = peso_trim = 0.0
            total = 1.0
        peso_media /= total
        peso_mediana /= total
        peso_moda /= total
        peso_trim /= total
        tendencia = (peso_media * media +
                     peso_mediana * mediana +
                     peso_moda * (0.0 if np.isnan(moda) else moda) +
                     peso_trim * (0.0 if np.isnan(trimmed) else trimmed))

    return pd.Series({
        "n": n,
        "media": media,
        "mediana": mediana,
        "moda": moda,
        "trimmed": trimmed,
        "MADN": madn(x),
        "bowley": bowley_asimetria,
        "exceso_kurtosis": g2,
        "s_mean_med": s_mean_med,
        "s_mode_med": s_mode_med if incluir_moda else np.nan,
        "s_mean_mode": s_mean_mode if incluir_moda else np.nan,
        "s_trim_med": s_trim_med if incluir_trimmed else np.nan,
        "s_mean_trim": s_mean_trim if incluir_trimmed else np.nan,
        "peso_media": peso_media,
        "peso_mediana": peso_mediana,
        "peso_moda": peso_moda,
        "peso_trimmed": peso_trim,
        "tendencia_ponderada": tendencia,
        "moda_robusta": es_moda_robusta,
        "altura_pico": altura_pico,
        "ancho_pico": ancho_pico
    })
