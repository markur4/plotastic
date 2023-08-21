#
# %%

import pandas as pd

from plotastic.dataframetool import DataFrameTool
from plotastic.statresults import StatResults
from plotastic.stattest import StatTest


# %% Class Assumptions


class Assumptions(StatTest):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)


# %%
