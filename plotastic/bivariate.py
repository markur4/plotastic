# !

import pingouin as pg

from plotastic.analysis import Analysis

# %%


class Bivariate(Analysis):
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)
