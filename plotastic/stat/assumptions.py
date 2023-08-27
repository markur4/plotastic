#
# %%

import pandas as pd

from plotastic.dimensions.dataframetool import DataFrameTool
from plotastic.stat.statresults import StatResults
from plotastic.stat.stattest import StatTest


# %% Class Assumptions


class Assumptions(StatTest):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)


# %%
