# !

import pingouin as pg

from plotastic.dataframetool import DataFrameTool

# %%


class Bivariate(DataFrameTool):
    def __init__(self, **dims_and_levels_kws):
        super().__init__(**dims_and_levels_kws)
