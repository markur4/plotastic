# !
# %% Imports

from typing import TYPE_CHECKING, Callable, Generator


import pyperclip
import pickle
from pathlib import Path


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


import markurutils as ut

from plotastic.dimensions.dataframetool import DataFrameTool

if TYPE_CHECKING:
    import numpy as np
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

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

    # ...__INIT__ ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

    # ... ITERATORS #...........................................................................................

    #
    # * NESTED / FLAT....................................#

    @property  # * [[ax11, ax12], [ax21, ax22]]
    def axes_nested(self) -> "np.ndarray":
        """Always returns a 2D nested array of axes, even if there is only one row or column."""
        if self.dims.row and self.dims.col:  # * both row and col
            return self.axes
        elif self.factors_is_1_facet:  # * either or
            return np.array([self.axes])
        else:  # * Single figure
            return np.array([self.axes]).reshape(1, 1)

    @property  # * [ax11, ax12, ax21, ax22]
    def axes_flat(self) -> "np.ndarray":
        """Always returns a 1D flattened array of axes, regardless of row, column, or single figure."""
        return self.axes_nested.flatten()

    #
    # * Associate with Keys ....................................#

    @property  # * >>> (R_lvl1, C_lvl1), ax11 >>> (R_lvl1, C_lv2), ax12 >>> (R_lvl2, C_lvl1), ax21 >> ...
    def axes_iter__keys_ax(self) -> Generator:
        """Returns: >> (R_lvl1, C_lvl1), ax11 >> (R_lvl1, C_lv2), ax12 >> (R_lvl2, C_lvl1), ax21 >> ..."""
        if self.factors_rowcol is None:
            # * If no row or col, return all axes and data
            yield None, self.axes  # ! Error for  df.groupby().get_group(None)
        else:
            for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
                yield key, ax

    @property
    def axes_dict(self) -> Generator:
        """Returns: {key: ax}"""
        return dict(self.axes_iter__keys_ax)

    #
    # * Associate with Rowcol   ................................#

    @property  # * >>> row_lvl1, (ax11, ax21, ...) >>> row_lvl2, (ax12, ax22, ...) >>> ...
    def axes_iter__row_axes(self):
        """Returns: row_lvl1, (ax11, ax21, ...) >> row_lvl2, (ax12, ax22, ...) >> ..."""
        for rowkey, axes in zip(self.levels_dict_dim["row"], self.axes_nested):
            yield rowkey, axes

    @property  # * >>> col_lvl1, (ax11, ax21, ...) >>> col_lvl2, (ax12, ax22, ...) >>> ...
    def axes_iter__col_axes(self):
        """Returns: col_lvl1, (ax11, ax21, ...) >> col_lvl2, (ax12, ax22, ...) >> ..."""
        for colkey, axes in zip(self.levels_dict_dim["col"], self.axes_nested.T):
            yield colkey, axes

    #
    # * Data  ..................................................#

    @property  # * >>> ax11, df11 >>> ax12, df12 >>> ax21, df21 >>> ...
    def axes_iter__ax_df(self):
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

            # ! old Version: Unelegant, but tested
            # for (key_ax, ax), (key_df, df) in zip(
            #     self.axes_iter__keys_ax, self.data_iter__key_facet
            # ):
            #     assert (
            #         key_df == key_ax
            #     ), f"Mismatch of dataframe_key and ax_key: {key_df} != {key_ax}"
            #     # * Seaborn breaks on Dataframes that are only NaNs
            #     if df[self.dims.y].isnull().all():
            #         continue
            #     else:
            #         yield ax, df

    #
    # * Selective   ............................................#

    @property  # * ax11 >>> ax12 >>> ax21 >>> ax22 >>> ...
    def axes_iter_leftmost_col(self):
        """Returns: >> ax11 >> ax21 >> ax31 >> ax41 >> ..."""
        for row in self.axes_nested:  # * Through rows
            yield row[0]  # * Leftmost column

    @property  # * >> axes excluding leftmost column
    def axes_iter_notleftmost_col(self):
        """Returns: >> axes excluding leftmost column"""
        for row in self.axes_nested:  # * Through rows
            for ax in row[1:]:  # * Through all columns except leftmost
                yield ax

    @property  # * ax31 >>> ax32 >>> ax33 >>> ax34 >>> ...
    def axes_iter_lowest_row(self):
        """Returns: >> ax31 >> ax32 >> ax33 >> ax34 >> ..."""
        for ax in self.axes_nested[-1]:  # * Pick Last row, iterate through columns
            yield ax

    @property  # * >> axes excluding lowest row
    def axes_iter_notlowest_row(self):
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
        hspace=None,
        width_ratios: list[int] = None,
        height_ratios: list[int] = None,
        figsize: tuple[int] = None,
        **subplot_kws: dict,
    ) -> "PlotTool | DataAnalysis":
        """Initialise matplotlib figure and axes objects

        Returns:
            tuple["mpl.figure.Figure", "mpl.axes.Axes"]: matplotlib figure and axes objects
        """

        # ... Handle kwargs
        ### Adds extra kwargs depending on kwargs already present
        wspace = 0.05 if sharey and (wspace is None) else wspace

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

        # ... SUBPLOTS
        self.fig, self.axes = plt.subplots(**KWS)

        # ... Edits
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

        Args:
            axes (mpl.axes.Axes): _description_
            kind (str, optional): _description_. Defaults to "strip".

        Returns:
            mpl.axes.Axes: _description_
        """
        # * If row or col, assure that axes_count == facet_count
        if self.factors_rowcol:
            assert (
                self.axes.flatten().size == self.len_rowlevels * self.len_collevels
            ), f"Axes size mismatch {self.axes.flatten().size} != {self.len_rowlevels * self.len_collevels}"

        ### Handle kwargs
        kws = dict()
        kws.update(sns_kws)

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
            # * Remove legend per axes, since we want one legend for the whole figure
            if self.dims.hue:
                ax.legend_.remove()  # ! also: legend=False doesn't work with sns.barplot for some reason..

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
    # ... Fig properties ............................................................................
    @property
    def figsize(self) -> tuple[int]:
        return self.fig.get_size_inches()

    #
    #
    # ... Small EDITS and those required to be set BEFORE seaborn plots......................................................
    def edit_axtitles_reset(self) -> "PlotTool | DataAnalysis":
        for key, ax in self.axes_iter__keys_ax:
            ax.set_title(self._standard_axtitle(key))
        return self

    def _edit_scale_base():
        pass
        

# ! # end class
# !
# !
# !
