# !

import pingouin as pg

from analysis import Analysis

# %%


class Bivariate(Analysis):
    def __init__(self, data, dims, verbose=False):
        super().__init__(data=data, dims=dims, verbose=verbose)
