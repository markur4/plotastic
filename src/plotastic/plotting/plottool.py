# !
# %% Imports



from typing import TYPE_CHECKING, Callable


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
        elif self.dims.row or self.dims.col:  # * either or
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
    def axes_iter__keys_ax(self):
        """Returns: >> (R_lvl1, C_lvl1), ax11 >> (R_lvl1, C_lv2), ax12 >> (R_lvl2, C_lvl1), ax21 >> ..."""
        if self.factors_rowcol is None:
            # * If no row or col, return all axes and data
            yield None, self.axes  # ! Error for  df.groupby().get_group(None)
        else:
            for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
                yield key, ax

    @property
    def axes_dict(self):
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
    def axes_iter_leftmost(self):
        """Returns: >> ax11 >> ax21 >> ax31 >> ax41 >> ..."""
        for row in self.axes_nested:
            yield row[0]

    @property  # * >> axes excluding leftmost column
    def axes_iter_notleftmost(self):
        """Returns: >> axes excluding leftmost column"""
        for row in self.axes_nested:
            for ax in row[1:]:
                yield ax

    @property  # * ax31 >>> ax32 >>> ax33 >>> ax34 >>> ...
    def axes_iter_lowerrow(self):
        """Returns: >> ax31 >> ax32 >> ax33 >> ax34 >> ..."""
        for ax in self.axes_nested[-1]:
            yield ax

    @property  # * >> axes excluding lowest row
    def axes_iter_notlowerrow(self):
        """Returns: >> axes excluding lowest row"""
        for row in self.axes_nested[:-1]:
            for ax in row:
                yield ax

    #
    #
    #  ... PLOT .........................................................................................................

    def subplots(
        self,
        wspace=0.4,
        hspace=0.7,
        width_ratios: list[int] = None,
        height_ratios: list[int] = None,
        figsize: tuple[int] = None,
        **subplot_kws: dict,
    ) -> "PlotTool":
        """Initialise matplotlib figure and axes objects

        Returns:
            tuple["mpl.figure.Figure", "mpl.axes.Axes"]: matplotlib figure and axes objects
        """
        ### Define standard kws
        KWS = dict(
            nrows=self.len_rowlevels,
            ncols=self.len_collevels,
            figsize=figsize,
            gridspec_kw=dict(
                wspace=wspace,
                hspace=hspace,
                width_ratios=width_ratios or [1] * self.len_collevels,
                height_ratios=height_ratios or [1] * self.len_rowlevels,
            ),
        )
        KWS = ut.update_dict_recursive(KWS, subplot_kws)

        ### SUBPLOTS
        self.fig, self.axes = plt.subplots(**KWS)

        ### Add titles to axes to provide basic orientation
        self.edit_axtitles_reset()

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

    def fillaxes(self, kind: str = "strip", **sns_kws: dict) -> "PlotTool":
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

    def plot(
        self, kind: str = "strip", subplot_kws: dict = None, **sns_kws
    ) -> "PlotTool":
        """Quick plotting, combines self.subplots and self.fillaxes its axes with seaborn graphics

        Args:
            kind (str, optional): _description_. Defaults to "strip".
            subplot_kws (dict, optional): _description_. Defaults to None.
            sns_kws (dict, optional): _description_. Defaults to None.

        Returns:
            fig_and_axes: _description_
        """
        ### Handle kwargs
        subplot_kws = subplot_kws or {}
        sns_kws = sns_kws or {}

        ### Standard kws for standard stripplot
        if kind == "strip" and len(sns_kws) == 0:
            sns_kws = dict(alpha=0.6, dodge=True)

        self.subplots(**subplot_kws)  # * Initialise Figure and Axes
        self.fillaxes(kind=kind, **sns_kws)  # * Fill axes with seaborn graphics
        self.edit_legend()  # * Add legend to figure
        plt.tight_layout()  # * Make sure everything fits nicely

        return self

    #
    #
    # ... Fig properties ............................................................................
    @property
    def figsize(self) -> tuple[int]:
        return self.fig.get_size_inches()

    #
    #
    # ... EDIT .........................................................................

    #
    # * Shapes & Sizes ........................................#
    # TODO: Until now, stick with the arguments supplied to self.subplots

    #
    # * Titles of axes .........................................#

    @staticmethod
    def _standard_axtitle(key: tuple[str] | str, connect="\n") -> str:
        """make axis title from key

        Args:
            key (tuple): _description_
        """
        if isinstance(key, str):
            return ut.capitalize(key)
        elif isinstance(key, tuple):
            keys = []
            for k in key:
                if isinstance(k, str):
                    keys.append(ut.capitalize(k))
                else:
                    keys.append(str(k))  # * Can't capitalize int
            return connect.join(keys)

    def edit_axtitles_reset(self) -> "PlotTool":
        for key, ax in self.axes_iter__keys_ax:
            ax.set_title(self._standard_axtitle(key))
        return self

    def edit_titles(
        self,
        axes: mpl.axes.Axes = None,
        axtitles: dict = None,
    ) -> "PlotTool":
        axes = axes or self.axes

        if not axtitles is None:
            for key, ax in self.axes_iter__keys_ax:
                ax.set_title(axtitles[key])
        return self

    def edit_titles_with_func(
        self,
        row_func: Callable = None,
        col_func: Callable = None,
        connect="\n",
    ) -> "PlotTool":
        """Applies formatting functions (e.g. lambda x: x.upper()) to row and col titles)"""
        row_func = row_func or (lambda x: x)
        col_func = col_func or (lambda x: x)

        for rowkey, axes in self.axes_iter__row_axes:
            for ax in axes:
                title = row_func(rowkey)
                ax.set_title(title)
        for colkey, axes in self.axes_iter__col_axes:
            for ax in axes:
                title = ax.get_title() + connect + col_func(colkey)
                ax.set_title(title)
        return self

    def edit_titles_with_func_SNIP(self) -> str:
        s = ""
        s += "row_format = lambda x: x #* e.g. try lambda x: x.upper() \n"
        s += "col_format = lambda x: x \n"
        s += "connect = '\\n' #* newline. Try ' | ' as a separator in the same line\n"
        s += "for rowkey, axes in DA.axes_iter__row_axes: \n"
        s += "\tfor ax in axes: \n"
        s += "\t\ttitle = row_format(rowkey) \n"
        s += "\t\tax.set_title(title) \n"
        s += "for colkey, axes in DA.axes_iter__col_axes: \n"
        s += "\tfor ax in axes: \n"
        s += "\t\ttitle = ax.get_title() + connect + col_format(colkey) \n"
        s += "\t\tax.set_title(title) \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    def edit_title_replace(self, titles: list) -> "PlotTool":
        """Edits axes titles. If list is longer than axes, the remaining titles are ignored

        Args:
            titles (list): Titles to be set. The order of the titles should be the same as the order of the axes, which is from left to right for row after row (like reading).

        Returns:
            PlotTool: The object itselt
        """

        for ax, title in zip(self.axes_flat, titles):
            ax.set_title(title)
        return self

    def edit_title_replace_SNIP(self):
        s = ""
        s += f"titles = {[ax.get_title() for ax in self.axes.flatten()]} \n"
        s += "for ax, title in zip(DA.axes.flatten(), titles): \n"
        s += "\tax.set_title(title) \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # * Labels of x- & y-axis  .................................#

    def edit_xy_axis_labels(
        self,
        leftmost: str = "",
        notleftmost: str = "",
        lowerrow: str = "",
        notlowerrow: str = "",
    ) -> "PlotTool":
        """Edits x- and y-axis labels

        Args:
            leftmost (str): Y-axis label for leftmost axes
            notleftmost (str): Y-axis label for not-leftmost axes
            lowerrow (str): x-axis label for lower row of axes
            notlowerrow (str): x-axis label for not-lower row of axes
        """
        leftmost = leftmost or self.dims.y
        notleftmost = notleftmost or ""
        lowerrow = lowerrow or self.dims.x
        notlowerrow = notlowerrow or ""

        ### y-axis labels
        for ax in self.axes_iter_leftmost:
            ax.set_ylabel(leftmost)
        for ax in self.axes_iter_notleftmost:
            ax.set_ylabel(notleftmost)

        ### x-axis labels
        for ax in self.axes_iter_lowerrow:
            ax.set_xlabel(lowerrow)
        for ax in self.axes_iter_notlowerrow:
            ax.set_xlabel(notlowerrow)
        return self

    def edit_xy_axis_labels_SNIP(self) -> str:
        s = ""
        s += "### y-axis labels \n"
        s += "for ax in DA.axes_iter_leftmost: \n"
        s += f"\tax.set_ylabel('{self.dims.y}') \n"
        s += "for ax in DA.axes_iter_notleftmost: \n"
        s += "\tax.set_ylabel('') \n"
        s += "### x-axis labels \n"
        s += "for ax in DA.axes_iter_lowerrow: \n"
        s += f"\tax.set_xlabel('{self.dims.x}') \n"
        s += "for ax in DA.axes_iter_notlowerrow: \n"
        s += "\tax.set_xlabel('') \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # * Scale of x- & y-axis  ..................................#

    def edit_y_scale_log(
        self, base=10, nonpositive="clip", subs=[2, 3, 4, 5, 6, 7, 8, 9]
    ) -> "PlotTool":
        for ax in self.axes_flat:
            ax.set_yscale(
                value="log",  # * "symlog", "linear", "logit", ...
                base=base,  # * Base of the logarithm
                nonpositive=nonpositive,  # * "mask": masked as invalid, "clip": clipped to a very small positive number
                subs=subs,  # * Where to place subticks between major ticks ! not working
            )

            # ax.yaxis.sety_ticks()
        return self

    def edit_x_scale_log(
        self, base=10, nonpositive="clip", subs=[2, 3, 4, 5, 6, 7, 8, 9]
    ) -> "PlotTool":
        for ax in self.axes.flatten():
            ax.set_xscale(
                value="log",  # * "symlog", "linear", "logit", ...
                base=base,  # * Base of the logarithm
                nonpositive=nonpositive,  # * "mask": masked as invalid, "clip": clipped to a very small positive number
                subs=subs,  # * Where to place subticks between major ticks ! not working
            )
        return self

    def edit_xy_scale_SNIP(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self._DOCS['set_xscale']} #\n"
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.set_yscale('log',  # * 'symlog', 'linear', 'logit',  \n"
        s += "\t\tbase=10,  \n"
        s += "\t\tnonpositive='clip', # * 'mask': masked as invalid, 'clip': clipped to a very small positive number \n"
        # ! s += "\t\tsubs=[2, 3, 4, 5], # * Where to place subticks between major ticks !! Removes both ticks and labels \n" \n"
        s += "\t) \n"
        s += "\t# ax.set_xscale('log') # ? Rescale x-axis\n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # * Ticks and their Labels .................................#

    def edit_y_ticklabel_percentage(
        self, decimals_major: int = 0, decimals_minor: int = 0
    ) -> "PlotTool":
        for ax in self.axes_flat:
            ax.yaxis.set_major_formatter(
                mpl.ticker.PercentFormatter(xmax=1, decimals=decimals_major)
            )
            ax.yaxis.set_minor_formatter(
                mpl.ticker.PercentFormatter(xmax=1, decimals=decimals_minor)
            )
        return self

    def edit_y_ticklabel_percentage_SNIP(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self._DOCS['percent_formatter']} #\n"
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0)) \n"
        s += "\tax.yaxis.set_minor_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=1)) \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    def edit_y_ticklabels_log_minor(self, subs: list = [2, 3, 5, 7]):
        """Displays minor ticklabels for log-scales. Only shows those ticks whose rounded mantissa (the digits from a float) is in subs

        Args:
            subs (list, optional): Mantissas (the digits from a float). Defaults to [2, 3, 5, 7].

        Returns:
            _type_: _description_
        """
        for ax in self.axes_flat:
            # * Set minor ticks, we need ScalarFormatter, others can't get casted into float
            ax.yaxis.set_minor_formatter(
                mpl.ticker.ScalarFormatter(useOffset=0, useMathText=False)
            )

            # * Iterate through labels
            for label in ax.yaxis.get_ticklabels(which="minor"):
                # ? How else to cast float from mpl.text.Text ???
                label_f = float(str(label).split(", ")[1])  # * Cast to float
                mantissa = int(
                    round(ut.mantissa_from_float(label_f))
                )  # * Calculate mantissa
                if not mantissa in subs:
                    label.set_visible(False)  # * Set those not in subs to invisible
        return self

    def edit_y_ticklabels_log_minor_SNIP(self) -> str:
        s = ""
        s += "for ax in DA.axes.flatten(): \n"
        s += "\t#* Set minor ticks, we need ScalarFormatter, others can't get casted into float \n"
        s += "\tax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter(useOffset=0, useMathText=False)) \n"
        s += "\t#* Iterate through labels \n"
        s += "\tfor label in ax.yaxis.get_ticklabels(which='minor'): \n"
        s += "\t\t# ? How else to cast float from mpl.text.Text ??? \n"
        s += "\t\tlabel_f = float(str(label).split(', ')[1])  #* Cast to float \n"
        s += "\t\tmantissa = int(round(ut.mantissa_from_float(label_f))) #* Calculate mantissa \n"
        s += "\t\tif not mantissa in [2, 3, 5, 7]: \n"
        s += "\t\t\tlabel.set_visible(False) # * Set those not in subs to invisible \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    def edit_x_ticklabels(
        self,
        lowerrow: list = None,
        notlowerrow: list = None,
        rotation: int = 0,
        ha: str = "center",
        va: str = "top",
        pad: float = 1,
        **text_kws,
    ) -> "PlotTool":
        """Edits x- ticklabels

        Args:
            lowerrow (list): x-axis ticklabels for lower row of axes
            notlowerrow (list): x-axis ticklabels for not-lower row of axes
            etc.
        """
        notlowerrow = notlowerrow or self.levels_dict_dim["x"]
        lowerrow = lowerrow or self.levels_dict_dim["x"]

        kws = dict(
            rotation=rotation,  # * Rotation in degrees
            ha=ha,  # * Horizontal alignment [ 'center' | 'right' | 'left' ]
            va=va,  # * Vertical Alignment   [ 'center' | 'top' | 'bottom' | 'baseline' ]
        )
        kws.update(text_kws)

        ticks = [i for i in range(len(lowerrow))]
        for ax in self.axes_iter_notlowerrow:
            ax.set_xticks(ticks=ticks, labels=notlowerrow, **kws)
            ax.tick_params(axis="x", pad=pad)
        for ax in self.axes_iter_lowerrow:
            ax.set_xticks(ticks=ticks, labels=lowerrow, **kws)
            ax.tick_params(axis="x", pad=pad)  # * Sets distance to figure
        return self

    def edit_x_ticklabels_SNIP(self) -> str:
        s = ""
        s += f"notlowerrow = {self.levels_dict_dim['x']} \n"
        s += f"lowerrow = {self.levels_dict_dim['x']} \n"
        s += "kws = dict( \n"
        s += "\trotation=0, #* Rotation in degrees \n"
        s += "\tha='center', #* Horizontal alignment [ 'center' | 'right' | 'left' ] \n"
        s += "\tva='top', #* Vertical Alignment   [ 'center' | 'top' | 'bottom' | 'baseline' ] \n"
        s += ") \n"
        s += f"ticks = {[i for i in range(len(self.levels_dict_dim['x']))]} \n"
        s += "for ax in DA.axes_iter_notlowerrow: \n"
        s += "\tax.set_xticks(ticks=ticks, labels=notlowerrow, **kws) \n"
        s += "\tax.tick_params(axis='x', pad=1) #* Sets distance to figure \n"
        s += "for ax in DA.axes_iter_lowerrow: \n"
        s += "\tax.set_xticks(ticks=ticks, labels=lowerrow, **kws) \n"
        s += "\tax.tick_params(axis='x', pad=1) #* Sets distance to figure \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # * Grid ...................................................#

    def edit_grid(self) -> "PlotTool":
        for ax in self.axes_flat:
            ax.yaxis.grid(True, which="major", ls="-", linewidth=0.5, c="grey")
            ax.yaxis.grid(True, which="minor", ls="-", linewidth=0.2, c="grey")
            ax.xaxis.grid(True, which="major", ls="-", linewidth=0.3, c="grey")
        return self

    def edit_grid_SNIP(self) -> str:
        s = ""
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.yaxis.grid(True, which='major', ls='-', linewidth=0.5, c='grey') \n"
        s += "\tax.yaxis.grid(True, which='minor', ls='-', linewidth=0.2, c='grey') \n"
        s += "\tax.xaxis.grid(True, which='major', ls='-', linewidth=0.3, c='grey') \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # * Legend .................................................#

    @property
    def legend_handles_and_labels(self):
        if (
            self.factors_rowcol
        ):  # * If we have row and col factors, we need to get the legend from the first axes
            handles, labels = self.axes.flatten()[0].get_legend_handles_labels()
        else:
            handles, labels = self.axes.get_legend_handles_labels()
        ### Remove duplicate handles from repeated plot layers
        by_label = dict(zip(labels, handles))
        handles = by_label.values()
        labels = by_label.keys()
        labels = [ut.capitalize(l) for l in labels]
        return handles, labels

    def edit_legend(self) -> "PlotTool":
        """Adds standard legend to figure"""
        self.fig.legend(
            title=ut.capitalize(self.dims.hue),
            handles=self.legend_handles_and_labels[0],
            labels=self.legend_handles_and_labels[1],
            loc="center right",
            bbox_to_anchor=(1.15, 0.50),
            borderaxespad=3,
            frameon=False,
            # fontsize=10, # ! overrides entry from rcParams
        )
        return self

    def edit_legend_SNIP(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self._DOCS['legend']} #\n"
        s += "DA.fig.legend( \n"
        s += f"\ttitle='{self.dims.hue}', #* Hue factor \n"
        s += "\thandles=DA.legend_handles_and_labels[0], #* If single axes, remove square brackets\n"
        s += "\tlabels=DA.legend_handles_and_labels[1], \n"
        s += "\tloc='center right', #* Rough location \n"
        s += "\tbbox_to_anchor=(1.15, 0.50), #* Exact location in width, height relative to complete figure \n"
        s += "\tncols=1, #* If >1, labels are displayed next to each other \n"
        s += "\tborderaxespad=3, #* Padding around axes, (pushing legend away) \n"
        s += "\tmarkerscale=1.5, #* Marker size relative to plotted datapoint \n"
        s += "\tframeon=False, #* Remove frame around legend \n"
        s += "\t# fontsize=10, #* Fontsize of legend labels \n"
        s += ")\n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # * fontsizes ..............................................#

    def edit_fontsizes(self, ticklabels=10, xylabels=10, axis_titles=10) -> "PlotTool":
        """Edits fontsizes in [pt]. Does not affect legent or suptitle

        Args:
            ticklabels (int, optional): _description_. Defaults to 9.
            xylabels (int, optional): _description_. Defaults to 10.
            axis_titles (int, optional): _description_. Defaults to 11.

        Returns:
            PlotTool: _description_
        """

        for ax in self.axes_flat:
            ax.tick_params(
                axis="y", which="major", labelsize=ticklabels
            )  # * Ticklabels
            ax.tick_params(axis="y", which="minor", labelsize=ticklabels - 1)
            ax.tick_params(axis="x", which="major", labelsize=ticklabels)
            ax.tick_params(axis="x", which="minor", labelsize=ticklabels - 1)
            ax.yaxis.get_label().set_fontsize(xylabels)  # * xy-axis labels
            ax.xaxis.get_label().set_fontsize(xylabels)
            ax.title.set_fontsize(axis_titles)  # * Title
        return self

    def edit_fontsizes_SNIP(self) -> str:
        s = ""
        s = "ticklabels, xylabels, axis_titles = 9, 10, 11 ### <--- CHANGE THIS [pt]\n"
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.tick_params(axis='y', which='major', labelsize=ticklabels) # * Ticklabels \n"
        s += "\tax.tick_params(axis='y', which='minor', labelsize=ticklabels-.5) \n"
        s += "\tax.tick_params(axis='x', which='major', labelsize=ticklabels) \n"
        s += "\tax.tick_params(axis='x', which='minor', labelsize=ticklabels-.5) \n"
        s += "\tax.yaxis.get_label().set_fontsize(xylabels) # * xy-axis labels\n"
        s += "\tax.xaxis.get_label().set_fontsize(xylabels) \n"
        s += "\tax.title.set_fontsize(axis_titles) # * Title\n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #


# ! # end class
# !
# !
# !
