# !

import pingouin as pg

from plotastic.dimsandlevels import DimsAndLevels

# %%


class Bivariate(DimsAndLevels):
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)
