# !
# %% Imports

from typing import TYPE_CHECKING, Callable, Generator, Tuple, Sequence


import pyperclip
import pickle
from pathlib import Path


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import seaborn as sns


import markurutils as ut

from plotastic.dimensions.dataframetool import DataFrameTool

if TYPE_CHECKING:
    # from numpy import ndarray
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

    # from matplotlib.axes import Axes # ! Doesn't work
    # import pandas as pd

    # import io

    # from matplotlib import axes


# %% Class: PlotTool ...................................................................................


class PlotTool(DataFrameTool):
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

    # ==__INIT__ :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
        fig, axes = plt.subplots(nrows=self.len_rowlevels, ncols=self.len_collevels)
        self.fig: mpl.figure.Figure = fig
        self.axes = axes
        plt.close()  # * Close the figure to avoid displaying it

        ### Buffer to store plot intermediates
        self.buffer = Path("__figcache")

    # __init__
    #
    #

    # == ITERATORS #..........................................................

    #
    ### NESTED / FLAT....................................#

    @property  # * [[ax11, ax12], [ax21, ax22]]
    def axes_nested(self) -> np.ndarray[np.ndarray[matplotlib.axes.Axes]]:
        """Always returns a 2D nested array of axes, even if there is only one row or column."""
        if bool(self.dims.row and self.dims.col):  # * both row and col
            return self.axes
        elif self.factors_is_1_facet:  # * either or
            return np.array([self.axes])  # * add one more layer
        else:  # * Single figure
            return np.array([self.axes]).reshape(1, 1)

    @property  # * [ax11, ax12, ax21, ax22]
    def axes_flat(self) -> Sequence[matplotlib.axes.Axes]:
        """Always returns a 1D flattened array of axes, regardless of row, column, or single figure."""
        # ! We need self.axes_nested, since axes is not always an array
        return self.axes_nested.flatten()

    #
    #### Associate with Keys ....................................#

    @property  # * >>> (R_lvl1, C_lvl1), ax11 >>> (R_lvl1, C_lv2), ax12 >>> (R_lvl2, C_lvl1), ax21 >> ...
    def axes_iter__keys_ax(
        self,
    ) -> Generator[Tuple[tuple | str, matplotlib.axes.Axes], None, None]:
        """Returns: >> (R_lvl1, C_lvl1), ax11 >> (R_lvl1, C_lv2), ax12 >> (R_lvl2, C_lvl1), ax21 >> ..."""
        if self.factors_rowcol is None:
            # * If no row or col, return all axes and data
            yield None, self.axes  # ! Error for  df.groupby().get_group(None)
        else:
            for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
                yield key, ax

    @property
    def axes_dict(self) -> dict:
        """Returns: {key: ax}"""
        return dict(self.axes_iter__keys_ax)

    #
    #### Associate with Rowcol   ................................#

    @property  # * >>> row_lvl1, (ax11, ax21, ...) >>> row_lvl2, (ax12, ax22, ...) >>> ...
    def axes_iter__row_axes(
        self,
    ) -> Generator[Tuple[str, matplotlib.axes.Axes], None, None]:
        """Returns: row_lvl1, (ax11, ax21, ...) >> row_lvl2, (ax12, ax22, ...) >> ..."""
        for rowkey, axes in zip(self.levels_dict_dim["row"], self.axes_nested):
            yield rowkey, axes

    @property  # * >>> col_lvl1, (ax11, ax21, ...) >>> col_lvl2, (ax12, ax22, ...) >>> ...
    def axes_iter__col_axes(
        self,
    ) -> Generator[Tuple[str, matplotlib.axes.Axes], None, None]:
        """Returns: col_lvl1, (ax11, ax21, ...) >> col_lvl2, (ax12, ax22, ...) >> ..."""
        for colkey, axes in zip(self.levels_dict_dim["col"], self.axes_nested.T):
            yield colkey, axes

    #
    ### Data  ..................................................#

    @property  # * >>> ax11, df11 >>> ax12, df12 >>> ax21, df21 >>> ...
    def axes_iter__ax_df(
        self,
    ) -> Generator[Tuple[matplotlib.axes.Axes, pd.DataFrame], None, None]:
        """Returns: >> (ax11, df11) >> (ax12, df12) >> (ax21, df21) >> ..."""
        if self.factors_rowcol is None:
            yield self.axes, self.data  # * If no row or col, return all axes and data
        else:
            # data_dict = self.data.groupby(self.factors_rowcol) # ! works too!
            for key in self.levelkeys_rowcol:
                ax = self.axes_dict[key]
                df = self.data_dict_skip_empty[key]
                # df = data_dict.get_group(key) # ! works, too!
                # ut.pp(df)
                # * Seaborn breaks on Dataframes that are only NaNs
                if df[self.dims.y].isnull().all():
                    continue
                else:
                    yield ax, df

    #
    #### Selective   ............................................#

    @property  # * ax11 >>> ax12 >>> ax21 >>> ax22 >>> ...
    def axes_iter_leftmost_col(self) -> Generator[matplotlib.axes.Axes, None, None]:
        """Returns: >> ax11 >> ax21 >> ax31 >> ax41 >> ..."""
        for row in self.axes_nested:  # * Through rows
            yield row[0]  # * Leftmost ax

    @property  # * >> axes excluding leftmost column
    def axes_iter_notleftmost_col(self) -> Generator[matplotlib.axes.Axes, None, None]:
        """Returns: >> axes excluding leftmost column"""
        for row in self.axes_nested:  # * Through rows
            for ax in row[1:]:  # * Through all columns except leftmost
                yield ax

    @property  # * ax31 >>> ax32 >>> ax33 >>> ax34 >>> ...
    def axes_iter_lowest_row(self) -> Generator[matplotlib.axes.Axes, None, None]:
        """Returns: >> ax31 >> ax32 >> ax33 >> ax34 >> ..."""
        if not self.dims.col is None:
            for ax in self.axes_nested[-1]:  # * Pick Last row, iterate through columns
                yield ax
        else:
            yield self.axes_flat[-1]  # * If no col, return last

    @property  # * >> axes excluding lowest row
    def axes_iter_notlowest_row(self) -> Generator[matplotlib.axes.Axes, None, None]:
        """Returns: >> axes excluding lowest row"""
        for row in self.axes_nested[:-1]:  # * All but last row
            for ax in row:  # * Through columns
                yield ax

    #
    #
    #  ... PLOT .........................................................................................................

    def subplots(
        self,
        sharey: bool = True,
        y_scale: str = None,
        y_scale_kws: dict = dict(),
        wspace=None,
        hspace=0.4,
        width_ratios: list[int] = None,
        height_ratios: list[int] = None,
        figsize: tuple[int] = None,
        **subplot_kws: dict,
    ) -> "PlotTool | DataAnalysis":
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
                width_ratios=width_ratios or [1] * self.len_collevels,  # * [1,1, ...]
                height_ratios=height_ratios or [1] * self.len_rowlevels,  # * [1,1, ...]
            ),
        )
        # KWS = ut.remove_None_recursive(KWS) # * Kick out Nones from dict
        # * User args override defaults
        KWS = ut.update_dict_recursive(KWS, subplot_kws)

        # == SUBPLOTS ==
        self.fig, self.axes = plt.subplots(**KWS)

        # == EDITS ==
        ### Add titles to axes to provide basic orientation
        self.edit_axtitles_reset()
        
        ### Scale
        # ! Must sometimes be done BEFORE seaborn functions, otherwise they might look weird
        if not y_scale is None:
            plt.yscale(y_scale, **y_scale_kws)
            
        return self

    def subplots_SNIP(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self._DOCS['plt.subplots']} #\n"
        s += "DA = DataAnalysis(data=DF, dims=DIMS)"
        s += f"""
        DA.fig, DA.axes = plt.subplots(
            nrows=DA.len_rowlevels, 
            ncols=DA.len_collevels,
            sharex=False, sharey=False, 
            figsize=(DA.len_collevels*2, DA.len_rowlevels*2), # * width, height in inches
            width_ratios={[1 for _ in range(self.len_collevels)]}, height_ratios={[1 for _ in range(self.len_rowlevels)]},
            gridspec_kw=dict(wspace=0.2, hspace=0.5),
            subplot_kw=None, 
            )\n"""
        s += "DA.reset_axtitles()  # * Add basic titles to axes"
        s = s.replace("        ", "")
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    def fillaxes(
        self, kind: str = "strip", **sns_kws: dict
    ) -> "PlotTool | DataAnalysis":
        """_summary_

        :param kind: _description_, defaults to "strip"
        :type kind: str, optional
        :return: _description_
        :rtype: PlotTool | DataAnalysis
        """
        
        ### If row or col, assure that axes_count == facet_count
        if self.factors_rowcol:
            assert (
                self.axes.flatten().size == self.len_rowlevels * self.len_collevels
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
        ):  # ! also: legend=False doesn't work with sns.barplot for some reason..
            self.remove_legend()

        return self

    def fillaxes_SNIP(self, kind: str = "strip", doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self._DOCS[kind]} #\n"
        s += "kws = dict(alpha=.8) \n"
        s += "for ax, df in DA.iter_axes_and_data: \n"
        s += f"\t{self._SNS_FUNCS_STR[kind]}(data=df, ax=ax, y='{self.dims.y}', x='{self.dims.x}', hue='{self.dims.hue}', **kws)\n"
        s += "\tax.legend_.remove() \n"  # * Remove legend, unless we want one per axes"
        s += "DA.legend()  # * Add one standard legend to figure \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    #
    # == Fig properties ............................................................................
    @property
    def figsize(self) -> tuple[int]:
        return self.fig.get_size_inches()

    #
    #
    # == Small EDITS and those required to be set BEFORE seaborn plots......................................................
    def edit_axtitles_reset(self) -> "PlotTool | DataAnalysis":
        for key, ax in self.axes_iter__keys_ax:
            ax.set_title(self._standard_axtitle(key))
        return self

    def remove_legend(self):
        for ax in self.axes_flat:
            if ax.legend_:
                ax.legend_.remove()


# ! # end class
# !
# !
# !
