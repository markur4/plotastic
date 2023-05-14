#

# %%

from cgi import test
from typing import Dict
from copy import copy, deepcopy

import pandas as pd

import markurutils as ut

from analysis import Analysis
from plothelper import PlotHelper
from plotsnippets import PlotSnippets

from stattester import StatTester
from assumptions import Assumptions
from omnibus import Omnibus
from posthoc import PostHoc

# %%


class DataAnalysis(PlotHelper, PlotSnippets, Assumptions, Omnibus, PostHoc, StatTester, Analysis):
    def __init__(
        self, data: pd.DataFrame, dims: dict, title: str = "untitled", verbose=True
    ):
        ### Inherit
        # * verbosity false, since each subclass can test its own DataFrame
        init_kws = dict(data=data, dims=dims, title=title, verbose=False)
        super().__init__(**init_kws)
        
        self.filer= ut.Filer(title=title)

        if verbose:
            self.warn_about_empties_and_NaNs()

        # self.plot = plot
        ### statistics
        # self.test = Test()


    # ... PLOTTING #.......................................................................................................

    def show_plot(self):
        pass
        # display(self.plot)


# %%
import markurutils as ut
import unittest


class TestDataAnalysis(unittest.TestCase):
    def test_switching(self):
        v = False
        data, dims = ut.load_dataset("tips", verbose=v)
        DA = DataAnalysis(data, dims, verbose=v)

        ### Chaining work?
        x, E1 = DA.dims.x, "size-cut"
        x_inchain, E2 = DA.switch("x", "hue", verbose=v).dims.x, "smoker"
        x_after_chaining, E3 = DA.dims.x, "size-cut"
        print(x, x_inchain, x_after_chaining)
        print(x != x_inchain)
        print(x == x_after_chaining)


if __name__ == "__main__":
    pass
    # unittest.main()


# %%
DF, dims = ut.load_dataset("tips")  # * Import Data
DA = DataAnalysis(data=DF, dims=dims, title="tips")  # * Make DataAnalysis Object

# %%
# DA.plot_data()

# %%


# %%
