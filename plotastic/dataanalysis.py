# !

# %% Imports

from cgi import test
from typing import Dict
from copy import copy, deepcopy

from pathlib import Path
import pickle

import pandas as pd

import markurutils as ut

# from analysis import Analysis

# from assumptions import Assumptions
from plotastic.multiplot import MultiPlot
from plotastic.omnibus import Omnibus
from plotastic.posthoc import PostHoc
from plotastic.bivariate import Bivariate

# from statresult import StatResult


# %% Class DataAnalysis


class DataAnalysis(MultiPlot, Omnibus, PostHoc, Bivariate):
    def __init__(
        self, data: pd.DataFrame, dims: dict, title: str = "untitled", verbose=True
    ):
        ### Inherit
        # * verbosity false, since each subclass can test its own DataFrame
        init_kws = dict(data=data, dims=dims, verbose=False)
        super().__init__(**init_kws)

        self._title = title
        self.filer = ut.Filer(title=title)

        if verbose:
            self.warn_about_empties_and_NaNs()

        # self.plot = plot
        ### statistics
        # self.test = Test()

    ### TITLE .......................................................................................................'''

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.filer.title = value

    def add_to_title(
        self, to_end: str = "", to_start: str = "", con: str = "_", inplace=False
    ) -> "Analysis":
        """
        :param to_start: str, optional (default="")
        String to add to start of title
        :param to_end: str, optional (default="")
        String to add to end of title
        :param con: str, optional (default="_")
        Conjunction-character to put between string addition and original title
        :return: str
        """
        a = self if inplace else ut.copy_by_pickling(self)

        if to_start:
            a.title = f"{to_start}{con}{a.title}"
        if to_end:
            a.title = f"{a.title}{con}{to_end}"
        return a

    # ... I/O PLOTTING #.............................................................................................

    def save_plot_tofile(
        self,
        g=None,
        name: str = None,
        filepath: Path | str = None,
        format=".pdf",
        overwrite=None,
        verbose=True,
    ):
        """

        :param g:
        :param filename: Provide custom full filepath. Overrides automatic filename generation. Will use default suffix from seaborn (.png)!
        :param filepath:
        :param format: file extension determines format
        :param overwrite:
        :return:
        """
        raise NotImplementedError
        g = g if g else self.graphic

        """### CONSTRUCT FILEPATH"""
        kind = self.kws.pre.get("kind", self.kws.func.__name__)
        default_filename = self.filer.get_filename(
            titlesuffix=kind
        )  # need to initialize
        if filepath:
            filepath = Path(filepath)
        else:
            filepath = self.filer.get_filepath(titlesuffix=kind)

        """OVERRIDE FILENAME"""
        if name:
            filepath = filepath.with_name(
                name
            )  # if filename.endswith(".pdf")  else f"{filename}.pdf"

        if not overwrite is None and not overwrite:
            filepath = mku.Filer.prevent_overwrite(
                filename=filepath.stem, parent=filepath.parent, ret_filepath=True
            )
        elif overwrite:
            filepath = filepath.with_name(
                str(self.filer.get_filename(titlesuffix=kind, overwrite=True))
            )

        """HANDLE FILE EXTENSION SUFFIX"""
        filepath = filepath.with_suffix(format)
        kws = dict()
        if filepath.suffix == ".pdf":
            kws["backend"] = "cairo"

        """SAVE IT"""
        backend = kws.get("backend", mpl.rcParams["backend"])
        if verbose:
            print(f"#! Saving '{filepath.name}'  in  '{filepath}' (backend={backend})")
        g.fig.savefig(filepath, bbox_inches="tight", facecolor="none", dpi=300, **kws)
        if verbose:
            print(f"#! ✅ {filepath.name} Saved! ")

        plt.close()

        return self

    def save_fig_tobuffer(self, name=""):
        filename = Path(self.buffer + name).with_suffix(".pickle")
        with open(filename, "wb") as file:
            pickle.dump((self.fig, self.axes), file)

    def load_fig_frombuffer(self, name=""):
        filename = Path(self.buffer + name).with_suffix(".pickle")
        with open(filename, "rb") as file:
            fig, axes = pickle.load(file)
        return (
            fig,
            axes,
        )  # ! can#t return the whole PlotTool object, since pyplot will mix the fig with previous objects

    # @staticmethod
    # def _redraw_fig(fig):
    #     """create a dummy figure and use its manager to display "fig" """
    #     dummy = plt.figure()  # * Make empty figure
    #     new_manager = dummy.canvas.manager  # * Get the figure's manager
    #     new_manager.canvas.figure = fig  # * Associate it with the figure
    #     fig.set_canvas(new_manager.canvas)
    #     return fig


# %% Import Test Data
DF, dims = ut.load_dataset("tips")  # * Import Data
DA = DataAnalysis(data=DF, dims=dims, title="tips")  # * Make DataAnalysis Object

# %% Catplot
DA.catplot()

# %% Unit Tests
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

        self.assertEqual(x, E1)
        self.assertEqual(x_inchain, E2)
        self.assertEqual(x_after_chaining, E3)


# %% __name__ == "__main__"

if __name__ == "__main__":
    import markurutils as ut
    import unittest

    # unittest.main()


# %%


# %%
