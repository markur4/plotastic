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

# from stattester import StatTester
from assumptions import Assumptions
from omnibus import Omnibus
from posthoc import PostHoc

# %%


class DataAnalysis(Analysis):
    def __init__(
        self, data: pd.DataFrame, dims: dict, title: str = "untitled", verbose=True
    ):
        ### Inherit
        # * verbosity false, since each subclass can test its own DataFrame
        init_kws = dict(data=data, dims=dims, verbose=False)
        super().__init__(**init_kws)

        ### Tools
        self.plothelper = PlotHelper(**init_kws)
        self.plotsnippets = PlotSnippets(**init_kws)
        self.assumptions = Assumptions(**init_kws)
        self.omnibus = Omnibus(**init_kws)
        self.posthoc = PostHoc(**init_kws)

        self.filer = ut.Filer(title=title, verbose=verbose)

        if verbose:
            self.warn_about_empties_and_NaNs()

        # self.plot = plot
        ### statistics
        # self.test = Test()

    # ... KEEP PARAMETERS SYNCED #.......................................................................................................

    def set_dims(self, dims):
        self.dims = dims
        self.plothelper.dims = dims
        self.plotsnippets.dims = dims
        self.assumptions.dims = dims
        self.omnibus.dims = dims
        self.posthoc.dims = dims
        return self

    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> "Analysis":
        # * NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply
        ### Make new dims
        dims = self.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)

        da = self if inplace else ut.copy_by_pickling(self)
        # * Keep all subclasses' dims synced
        da = da.set_dims(dims)

        return da

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

        ### Subclasses in sync?
        x_ph, E4 = DA.plothelper.dims.x, "size-cut"
        x_ph_inchain, E5 = DA.switch("x", "hue", verbose=v).plothelper.dims.x, "smoker"
        x_ph_after_chaining, E6 = DA.plothelper.dims.x, "size-cut"
        print(x_ph, x_ph_inchain, x_ph_after_chaining)
        print(x_ph != x_ph_inchain)
        print(x_ph == x_ph_after_chaining)

        self.assertEqual(x, E1)
        self.assertEqual(x_inchain, E2)
        self.assertEqual(x_after_chaining, E3)
        self.assertEqual(x_ph, E4)
        self.assertEqual(x_ph_inchain, E5)
        self.assertEqual(x_ph_after_chaining, E6)


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
