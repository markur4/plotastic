# !!

# %% Imports
from typing import TYPE_CHECKING

from copy import deepcopy

# from typing import Self # !! only for python 3.11. Not really needed, since "DataAnalysis" as typehint works with vscode

from pathlib import Path
import pickle

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from plotastic import docstrings

# import markurutils as ut
import plotastic.utils.utils as ut
from plotastic.dataanalysis.annotator import Annotator
from plotastic.dataanalysis.filer import Filer

# from statresult import StatResult
if TYPE_CHECKING:
    import matplotlib as mpl
    from matplotlib.transforms import Bbox

# %% Class DataAnalysis


class DataAnalysis(Annotator):
    # == __init__ ======================================================
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        subject: str = None,
        levels: list[tuple[str]] = None,
        title: str = "untitled",
        verbose=True,
    ) -> "DataAnalysis":
        ### Inherit
        # !! verbosity set to False, since each subclass shouldn't test its own DataFrame
        dataframetool_kws = dict(
            data=data,
            dims=dims,
            subject=subject,
            levels=levels,
        )
        super().__init__(**dataframetool_kws)

        self._title = title
        self.filer = Filer(title=title)

        if verbose:
            self.data_check_integrity()

        # self.plot = plot
        ### statistics
        # self.test = Test()

    # ==
    # == TITLE =========================================================

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.filer.title = value

    def title_add(
        self,
        to_end: str = "",
        to_start: str = "",
        con: str = "_",
        inplace=False,
    ) -> "DataAnalysis":
        """Adds string to start and/or end of title


        :param to_start: str, optional (default="")
        String to add to start of title
        :param to_end: str, optional (default="")
        String to add to end of title
        :param con: str, optional (default="_")
        Conjunction-character to put between string addition and original title
        :return: str
        """
        a: "DataAnalysis" = self if inplace else deepcopy(self)

        if to_start:
            a.title = f"{to_start}{con}{a.title}"
        if to_end:
            a.title = f"{a.title}{con}{to_end}"
        return a

    # ==
    # == Saving stuff ==================================================

    @docstrings.subst(param_overwrite=docstrings.param_overwrite)
    def save_statistics(
        self,
        fname: str = "plotastic_results",
        overwrite: str | bool = "day",
    ) -> None:
        """Exports all statistics to one excel file. Different sheets for different
        tests

        :param overwrite: {param_overwrite}
        :param out: Path to save excel file, optional (default="")
        :type out: str, optional
        """

        ### Overwrite Protection
        if (not overwrite and not overwrite is None) or isinstance(
            overwrite, str
        ):
            fname = self.filer.prevent_overwrite(
                fname=fname, overwrite=overwrite
            )

        ### Save Statistics
        self.results.save(fname=fname)

    # @docstrings.subst(param_overwrite=docstrings.param_overwrite)
    # def save_fig(
    #     self,
    #     fname: str | Path = "plotastic_results",
    #     format: str = "pdf",
    #     fig: Figure = None,
    #     overwrite: str | bool = "day",  #' Added overwrite protection
    #     dpi: int | str = 300,  # !! mpl default is "figure"
    #     bbox_inches: "str | Bbox" = "tight",
    #     pad_inches: float = 0.1,
    #     facecolor: str = "none",  # !! mpl default is "auto", using current figure facecolor
    #     edgecolor: str = "none",  # !! mpl default is "auto", using current figure edgecolor
    #     backend: str = None,
    #     **user_kwargs,
    # ) -> "DataAnalysis":
    #     """Calls plt.figure.Figure.savefig(). Also provides an overwrite protection

    #     {param_overwrite}
    #     :param fname: A path, or a Python file-like object. If format is set, it
    #         determines the output format, and the file is saved as fname. Note that
    #         fname is used verbatim, and there is no attempt to make the extension, if
    #         any, of fname match format, and no extension is appended.

    #         If format is not set, then the format is inferred from the extension of
    #         fname, if there is one. If format is not set and fname has no extension,
    #         then the file is saved with rcParams["savefig.format"] (default: 'png') and
    #         the appropriate extension is appended to fname., defaults to
    #         "plotastic_results"
    #     :type fname: str | path.Path, optional
    #     :param format: The file format, e.g. 'png', 'pdf', 'svg', ... The behavior when
    #         this is unset is documented under fname., defaults to "pdf"
    #     :type format: str, optional
    #     :param dpi: The resolution in dots per inch. If 'figure', use the figure's dpi
    #         value., defaults to 300
    #     :type dpi: int, optional
    #     :param bbox_inches: Bounding box in inches: only the given portion of the figure
    #         is saved. If 'tight', try to figure out the tight bbox of the figure.,
    #         defaults to "tight"
    #     :type bbox_inches: str | plt.Bbox, optional
    #     :param pad_inches: Amount of padding in inches around the figure when
    #         bbox_inches is 'tight'. If 'layout' use the padding from the constrained or
    #         compressed layout engine; ignored if one of those engines is not in use,
    #         defaults to 0.1
    #     :type pad_inches: float, optional
    #     :param facecolor: The facecolor of the figure. If 'auto', use the current figure
    #         facecolor., defaults to "auto"
    #     :type facecolor: str, optional
    #     :param edgecolor: The edgecolor of the figure. If 'auto', use the current figure
    #         edgecolor., defaults to "auto"
    #     :type edgecolor: str, optional
    #     :param backend: The backend to use for the rendering. If None, use
    #         rcParams["savefig.backend"], otherwise use backend, defaults to None
    #     :type backend: str, optional
    #     :param user_kwargs: Additional kwargs passed to plt.figure.Figure.savefig()
    #     """

    #     ### Gather arguments
    #     kwargs = dict(
    #         # fname=self.title, # !! pass it directly
    #         format=format,
    #         dpi=dpi,
    #         bbox_inches=bbox_inches,
    #         pad_inches=pad_inches,
    #         facecolor=facecolor,
    #         edgecolor=edgecolor,
    #         backend=backend,
    #     )
    #     kwargs.update(**user_kwargs)  #' Add user kwargs

    #     ### Overwrite protection
    #     if (not overwrite and not overwrite is None) or isinstance(overwrite, str):
    #         fname = self.filer.prevent_overwrite(filename=fname, mode=overwrite)

    #     ### Add Suffix
    #     fname = Path(fname).with_suffix("." + format)

    #     ### take figure
    #     if fig is None:
    #         fig = self.fig
    #     fig.savefig(fname, **kwargs)

    #     ### Save figure
    #     #  Not working, self.fig is never updated during plotting (only axes?)
    #     # self.fig.savefig(fname, **kwargs)
    #     # plt.savefig(fname, **kwargs)

    #     return self

    # def save_all(
    #     self,
    #     fname: str = "plotastic_results",
    #     overwrite: str | bool = "day",
    #     savefig_kws: dict = None,
    # ) -> None:
    #     """Exports all files stored in DataAnalysis object

    #     :param fname: Path to save excel file, optional (default="")
    #     :type fname: str, optional
    #     :param overwrite: Mode of overwrite protection. If "day", it simply adds the
    #         current date at the end of the filename, causing every output on the same
    #         day to overwrite itself. If "nothing" ["day", "nothing"], files with the
    #         same filename will be detected in the current work directory and a number
    #         will be added to the filename. If True, everything will be overwritten.,
    #         defaults to "day"
    #     :type overwrite: str|bool, optional
    #     :param savefig_kws: Additional kwargs passed to plt.figure.Figure.savefig()
    #     :type savefig_kws: dict, optional
    #     """

    #     ### Gather Arguments
    #     if savefig_kws is None:
    #         savefig_kws = dict()

    #     # if (not overwrite and not overwrite is None) or isinstance(overwrite, str):
    #     #     fname = self.filer.prevent_overwrite(filename=fname, mode=overwrite)

    #     self.save_statistics(fname=fname, overwrite=overwrite)
    #     self.save_fig(fname=fname, overwrite=overwrite, **savefig_kws)

    # @staticmethod
    # def _redraw_fig(fig):
    #     """create a dummy figure and use its manager to display "fig" """
    #     dummy = plt.figure()  #' Make empty figure
    #     new_manager = dummy.canvas.manager  #' Get the figure's manager
    #     new_manager.canvas.figure = fig  #' Associate it with the figure
    #     fig.set_canvas(new_manager.canvas)
    #     return fig


# %%
if __name__ == "__main__":
    from plotastic.example_data.load_dataset import load_dataset

    DF, dims = load_dataset("qpcr")
    DA = DataAnalysis(DF, dims)

    # %% Fill DA with stuff


# %%
