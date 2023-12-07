# !!
# %% Imports

from typing import (
    TYPE_CHECKING,
    # Callable,
    Generator,
    Tuple,
    Sequence,
    # TypeVar,
    # Generic,
)


# import pyperclip

# import pickle
from pathlib import Path


import numpy as np
import numpy.typing as npt
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

import plotastic.utils.utils as ut

from plotastic.dimensions.dataintegrity import DataIntegrity

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis


# %%

# ? HOW DO WE TYPE HINT np.ndarray FILLED WITH plt.Axes ?????
# ? Tried TypeVar, doesn't work
if __name__ == "__main__":
    # axes: Annotated[np.ndarray[plt.Axes], "axes"]
    import matplotlib

    axes: npt.NDArray[matplotlib.axes.Axes]

    for ax in axes:
        for a in ax:
            # a: plt.Axes # ? Explicit type hinting works, but not feasable !
            a.set_title("test")  # ? Not detected by VScode !!!


# %% Class: PlotToolf


class SubPlot(DataIntegrity):
    _SNS_FUNCS = {
        "bar": sns.barplot,
        "point": sns.pointplot,
        "strip": sns.stripplot,
        "box": sns.boxplot,
        "violin": sns.violinplot,
        "swarm": sns.swarmplot,
        "boxen": sns.boxenplot,
        "count": sns.countplot,
        "hist": sns.histplot,
        "kde": sns.kdeplot,
        # "ecdf": sns.ecdfplot,
        "rug": sns.rugplot,
        "line": sns.lineplot,
        "rel": sns.relplot,
    }

    _SNS_FUNCS_STR = {
        "bar": "sns.barplot",
        "point": "sns.pointplot",
        "strip": "sns.stripplot",
        "box": "sns.boxplot",
        "violin": "sns.violinplot",
        "swarm": "sns.swarmplot",
        "boxen": "sns.boxenplot",
        "count": "sns.countplot",
        "hist": "sns.histplot",
        "kde": "sns.kdeplot",
        # "ecdf": "sns.ecdfplot",
        "rug": "sns.rugplot",
        "line": "sns.lineplot",
        "rel": "sns.relplot",
    }

    _DOCS = {
        ### plt functions
        "plt.subplots": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots",
        "set_xscale": "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale",
        "legend": "https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.legend",
        "ticker": "https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker",
        "percent_formatter": "https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.PercentFormatter",
        ### sns functions
        "bar": "https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot",
        "point": "https://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot",
        "strip": "https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot",
        "box": "https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot",
        "violin": "https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot",
        "swarm": "https://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot",
        "boxen": "https://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot",
        "count": "https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot",
        "hist": "https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot",
        "kde": "https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot",
        # "sns.ecdfplot": "https://seaborn.pydata.org/generated/seaborn.ecdfplot.html#seaborn.ecdfplot",
        "rugt": "https://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot",
        "line": "https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot",
        "rel": "https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot",
    }

    # ==__INIT__ =======================================================

    def __init__(self, **dataframetool_kws):
        """
        _summary_

        Args:
            data (pd.DataFrame): _description_
            dims (dict): _description_
            verbose (bool, optional): _description_. Defaults to False.
            fig (mpl.figure.Figure, optional): _description_. Defaults to None.
            axes (mpl.axes.Axes, optional): _description_. Defaults to None.

        Returns:
            PlotTool: _description_
        """
        ### Inherit from DataFrameTool
        super().__init__(**dataframetool_kws)

        ### Initialise figure and axes
        fig, axes = plt.subplots(
            nrows=self.len_rowlevels, ncols=self.len_collevels
        )
        self.fig: plt.Figure = fig
        self.axes: plt.Axes | np.ndarray[plt.Axes] = axes
        plt.close()  #' Close the figure to avoid displaying it

    # __init__
    #
    #

    # == ITERATORS =====================================================
    #
    ### NESTED / FLAT....................................#

    @property  #' [[ax11, ax12], [ax21, ax22]]
    def axes_nested(self) -> np.ndarray[np.ndarray[plt.Axes]]:
        """Always returns a 2D nested array of axes, even if there is
        only one row or column."""

        # !! subplots(square=True) returns a 2D array, so no need to reshape
        # !! BUT that would mean I'd also have to refactor everything that needs flat.
        # !! So let's just keep it this way
        # return self.axes

        if bool(self.dims.row and self.dims.col):  #' both row and col
            return self.axes
        elif self.factors_is_1_facet:  #' either or
            return np.array([self.axes])  #' add one more layer
        else:  #' Single figure
            return np.array([self.axes]).reshape(1, 1)

    @property  #' [ax11, ax12, ax21, ax22]
    def axes_flat(self) -> Sequence[plt.Axes]:
        """Always returns a 1D flattened array of axes, regardless of
        row, column, or single figure."""
        # !! We need self.axes_nested, since axes is not always an array
        return self.axes_nested.flatten()

    #
    #### Associate with Keys ....................................#

    @property  #' >>> (R_lvl1, C_lvl1), ax11 >>> (R_lvl1, C_lv2), ax12 >>> (R_lvl2, C_lvl1), ax21 >> ...
    def axes_iter__keys_ax(
        self,
    ) -> Generator[Tuple[tuple | str, plt.Axes], None, None]:
        """Returns: >> (R_lvl1, C_lvl1), ax11 >> (R_lvl1, C_lv2), ax12
        >> (R_lvl2, C_lvl1), ax21 >> ..."""
        if self.factors_rowcol is None:
            #' If no row or col, return all axes and data
            yield None, self.axes  # !! Error for  df.groupby().get_group(None)
        else:
            for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
                yield key, ax

    @property
    def axes_dict(self) -> dict:
        """Returns: {key: ax}"""
        return dict(self.axes_iter__keys_ax)

    #
    #### Associate with Rowcol   ................................#

    @property  #' >>> row_lvl1, (ax11, ax21, ...) >>> row_lvl2, (ax12, ax22, ...) >>> ...
    def axes_iter__row_axes(
        self,
    ) -> Generator[Tuple[str, plt.Axes], None, None]:
        """Returns: row_lvl1, (ax11, ax21, ...) >> row_lvl2, (ax12,
        ax22, ...) >> ..."""
        axes: plt.Axes
        for rowkey, axes in zip(self.levels_dict_dim["row"], self.axes_nested):
            yield rowkey, axes

    @property  #' >>> col_lvl1, (ax11, ax21, ...) >>> col_lvl2, (ax12, ax22, ...) >>> ...
    def axes_iter__col_axes(
        self,
    ) -> Generator[Tuple[str, plt.Axes], None, None]:
        """Returns: col_lvl1, (ax11, ax21, ...) >> col_lvl2, (ax12,
        ax22, ...) >> ..."""
        for colkey, axes in zip(
            self.levels_dict_dim["col"], self.axes_nested.T
        ):
            yield colkey, axes

    #
    ### Data  ..................................................#

    @property  #' >>> ax11, df11 >>> ax12, df12 >>> ax21, df21 >>> ...
    def axes_iter__ax_df(
        self,
    ) -> Generator[Tuple[plt.Axes, pd.DataFrame], None, None]:
        """Returns: >> (ax11, df11) >> (ax12, df12) >> (ax21, df21) >> ..."""
        if self.factors_rowcol is None:
            yield self.axes, self.data  #' If no row or col, return all axes and data
        else:
            # data_dict = self.data.groupby(self.factors_rowcol) # !! works too!
            for key in self.levelkeys_rowcol:
                ax = self.axes_dict[key]
                df = self.data_dict_skip_empty[key]
                # df = data_dict.get_group(key) # !! works, too!
                # ut.pp(df)
                #' Seaborn breaks on Dataframes that are only NaNs
                if df[self.dims.y].isnull().all():
                    continue
                else:
                    yield ax, df

    #
    #### Selective   ............................................#

    @property  #' ax11 >>> ax12 >>> ax21 >>> ax22 >>> ...
    def axes_iter_leftmost_col(self) -> Generator[plt.Axes, None, None]:
        """Returns: >> ax11 >> ax21 >> ax31 >> ax41 >> ..."""
        if self.dims.col:
            for row in self.axes_nested:  #' Through rows
                yield row[0]  #' Leftmost ax
        else:
            for ax in self.axes_flat:
                yield ax

    @property  #' >> axes excluding leftmost column
    def axes_iter_notleftmost_col(self) -> Generator[plt.Axes, None, None]:
        """Returns: >> axes excluding leftmost column"""
        if self.dims.col:
            for row in self.axes_nested:  #' Through rows
                for ax in row[1:]:  #' Through all columns except leftmost
                    yield ax

    @property  #' ax31 >>> ax32 >>> ax33 >>> ax34 >>> ...
    def axes_iter_lowest_row(self) -> Generator[plt.Axes, None, None]:
        """Returns: >> ax31 >> ax32 >> ax33 >> ax34 >> ..."""
        if self.dims.col:
            for ax in self.axes_nested[-1]:
                #' Pick Last row, iterate through columns
                yield ax
        else:
            yield self.axes_flat[-1]  #' If no col, return last

    @property  #' >> axes excluding lowest row
    def axes_iter_notlowest_row(self) -> Generator[plt.Axes, None, None]:
        """Returns: >> axes excluding lowest row"""
        for row in self.axes_nested[:-1]:  #' All but last row
            for ax in row:  #' Through columns
                yield ax

    # ==
    # ==
    # ==   PLOT  =======================================================

    def subplots(
        self,
        sharey: bool = True,
        y_scale: str = None,
        y_scale_kws: dict = dict(),
        wspace=None,
        hspace=0.8,
        width_ratios: list[int] = None,
        height_ratios: list[int] = None,
        figsize: tuple[int] = None,
        **subplot_kws: dict,
    ) -> "SubPlot | DataAnalysis":
        """Initialise matplotlib figure and axes objects

        Returns:
            tuple["mpl.figure.Figure", "mpl.axes.Axes"]: matplotlib figure and axes objects
        """

        # == Handle kwargs
        ### Adds extra kwargs depending on kwargs already present
        if sharey and (wspace is None):
            wspace = 0.05
        elif not sharey and (wspace is None):
            wspace = 0.5

        ### Redirect kwargs, provide function defaults, flatten access
        KWS = dict(
            sharey=sharey,
            nrows=self.len_rowlevels,
            ncols=self.len_collevels,
            figsize=figsize,
            gridspec_kw=dict(
                wspace=wspace,
                hspace=hspace,
                width_ratios=width_ratios
                or [1] * self.len_collevels,  #' [1,1, ...]
                height_ratios=height_ratios
                or [1] * self.len_rowlevels,  #' [1,1, ...]
            ),
        )
        # KWS = ut.remove_None_recursive(KWS) #' Kick out Nones from dict
        #' User args override defaults
        KWS = ut.update_dict_recursive(KWS, subplot_kws)

        ### SUBPLOTS ===========
        self.fig: plt.Figure
        self.axes: plt.Axes | np.ndarray[plt.Axes]
        self.fig, self.axes = plt.subplots(
            # squeeze=False, # !! Always return 2D array. Don't use, requires unnecessary refactoring
            **KWS,
        )

        self.fig.subplots_adjust(hspace=1)

        ### Edits
        ### Add titles to axes to provide basic orientation
        self.edit_axtitles_reset()

        ### Scale
        # !! Must sometimes be done BEFORE seaborn functions, otherwise they might look weird
        if not y_scale is None:
            plt.yscale(y_scale, **y_scale_kws)

        ### Save current plot as attributes
        # ? Not needed? Further actions edit self.axes reference
        # self.fig = fig  #' plt.Figure
        # self.axes = ax  #' np.ndarray[np.ndarray[plt.Axes]]

        return self

    def fillaxes(
        self, kind: str = "strip", **sns_kws: dict
    ) -> "SubPlot | DataAnalysis":
        """Iterates through self.data and self.axes and plots data into
        axes using seaborn plotting functions.

        :param kind: Kind of plot, any seaborn plot should work ["bar",
            "box", "strip", "swarm", "point", "violin", etc.]. ,
            defaults to "strip"
        :type kind: str, optional
        :param sns_kws: Keyword arguments passed to seaborn function
            selected in kind.
        :return: DataAnalysis object for method chaining
        :rtype: PlotTool | DataAnalysis
        """

        ### If row or col, assure that axes_count == facet_count
        if self.factors_rowcol:
            assert (
                self.axes.flatten().size
                == self.len_rowlevels * self.len_collevels
            ), f"Axes size mismatch {self.axes.flatten().size} != {self.len_rowlevels * self.len_collevels}"

        ### Handle kwargs
        kws = dict()
        kws.update(sns_kws)

        # == PLOT ==
        ### Iterate through data and axes
        for ax, df in self.axes_iter__ax_df:
            self._SNS_FUNCS[kind](
                data=df,
                ax=ax,
                y=self.dims.y,
                x=self.dims.x,
                hue=self.dims.hue,
                **kws,
            )

        # == EDITS ==
        ### Remove y-labels from all but leftmost column
        if not self.dims.col is None:
            for ax in self.axes_iter_notleftmost_col:
                ax.set_ylabel("")

        ### Remove legend per axes, since we want one legend for the whole figure
        if (
            self.dims.hue
        ):  # !! also: legend=False doesn't work with sns.barplot for some reason..
            self.remove_legend()

        ### TODO (Save current plot as attribute, so that we can save mid-plot
        # self.fig = plt.gcf() # !! Doesn't work, seems to overwrite self.axes to list
        # self.axes = self.fig.get_axes() # !! Doesn't work, always returns list, not 2D
        # array

        ### Explicitly update the figure to include modified axes
        # plt.figure(self.fig.number) # !! from chatgpt doesn't work

        ### Update the figure reference
        # ax = self.axes_flat[-1]  # Select the last modified axes
        # self.fig = ax.get_figure()  # Update the figure reference

        return self

    #
    #
    # == Fig properties ================================================

    @property
    def figsize(self) -> tuple[int]:
        return self.fig.get_size_inches()

    #
    #
    # == Small EDITS and those required to be set BEFORE seaborn plots =
    def edit_axtitles_reset(self) -> "SubPlot | DataAnalysis":
        for key, ax in self.axes_iter__keys_ax:
            ax.set_title(self._standard_axtitle(key))
        return self

    def remove_legend(self):
        for ax in self.axes_flat:
            if ax.legend_:
                ax.legend_.remove()

    # == Save ==========================================================
    # # !! NOT WORKING, just use plt.savefig() manually, I couldn't figure this out

    # def save_fig(self, **savefig_kwargs) -> "PlotTool | DataAnalysis":
    #     """Calls plt.figure.Figure.savefig(). Overridden by DataAnalysis.save_fig(), but
    #     useful to have here for testing purposes..?

    #     :param safefig_kwargs: kwargs passed to plt.figure.Figure.savefig()
    #     """
    #     # !! This function is overriden by DataAnalysis.save_fig()
    #     plt.savefig(**savefig_kwargs)
    #     # !! Not working, self.fig is Never updated during .fillaxes!
    #     # self.fig.savefig(**savefig_kwargs)
    #     return self

    # == Buffer ========================================================
    # !! Not used, can't get full control over matplotlib hidden objects

    # @staticmethod
    # def save_fig_tobuffer(fig: Figure, axes: np.ndarray, name=""):
    #     buffer = Path("__figcache" + name)
    #     filename = Path(buffer).with_suffix(".pickle")
    #     with open(filename, "wb") as file:
    #         pickle.dump((fig, axes), file)

    # @staticmethod
    # def load_fig_frombuffer(fig: Figure, axes: np.ndarray, name=""):
    #     buffer = Path("__figcache" + name)
    #     filename = Path(buffer).with_suffix(".pickle")
    #     with open(filename, "rb") as file:
    #         fig, axes = pickle.load(file)
    #     # !! can't return the whole PlotTool object, since pyplot will mix the fig with previous objects
    #     return fig, axes

    ### Buffer to store plot intermediates
