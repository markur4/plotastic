# !!

import pingouin as pg

from plotastic.dimensions.dataframetool import DataFrameTool

# %%


class Bivariate(DataFrameTool):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)
