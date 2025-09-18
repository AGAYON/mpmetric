import pandas as pd
from pandas.api.extensions import register_series_accessor
from .api import metrica1, metrica2, metrica1_plot, metrica2_plot, diagnostic_plot

@register_series_accessor("mp")
class MPAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def metrica1(self, **params):
        return metrica1(self._obj, **params)

    def metrica2(self, **params):
        return metrica2(self._obj, **params)

    def metrica1_plot(self, **params):
        return metrica1_plot(self._obj, **params)

    def metrica2_plot(self, **params):
        return metrica2_plot(self._obj, **params)

    def diagnostic_plot(self, **params):
        return diagnostic_plot(self._obj, **params)
