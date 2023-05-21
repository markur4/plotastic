# !
# %% Imports

from __future__ import annotations

# import decimal
# from distutils.fancy_getopt import WS_TRANS
# from keyword import kwlist
# from shutil import which
# from turtle import width
from turtle import left
from typing import TYPE_CHECKING, Callable
from pyparsing import col

# from click import edit

import pyperclip

# from matplotlib import axes
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display


import markurutils as ut
from analysis import Analysis

if TYPE_CHECKING:
    import numpy as np

    # from matplotlib import axes


# %% Class: PlotTool ...................................................................................


class PlotTool(Analysis):
    SNS_FUNCS = {
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

    SNS_FUNCS_STR = {
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

    DOCS = {
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

    # ...__INIT__ .....................

    def __init__(self, data: pd.DataFrame, dims: dict, verbose=False) -> PlotTool:
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
        ### Inherit from Analysis
        super().__init__(data=data, dims=dims, verbose=verbose)

        ### Initialise figure and axes
        fig, axes = plt.subplots(nrows=self.len_rowlevels, ncols=self.len_collevels)
        self.fig: mpl.figure.Figure = fig
        self.axes = axes
        plt.close()  # * Close the figure to avoid displaying it

    #
    #
    #

    # ... handle shape of axes

    @property
    def axes_nested(self) -> np.ndarray:
        """Always returns a 2D nested array of axes, even if there is only one row or column."""
        if self.dims.row and self.dims.col:  # * both row and col
            return self.axes
        elif self.dims.row or self.dims.col:  # * either or
            return np.array([self.axes])
        else:  # * Single figure
            return np.array([self.axes]).reshape(1, 1)

    # ... ITERATORS #...........................................................................................

    @property
    def axes_iter__keys_ax(self):
        """Returns: >> (R_lvl1, C_lvl1), ax11 >> (R_lvl1, C_lv2), ax12 >> (R_lvl2, C_lvl1), ax21 >> ..."""
        if self.factors_rowcol is None:
            # * If no row or col, return all axes and data
            yield None, self.axes  # ! Error for  df.groupby().get_group(None)
        else:
            for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
                yield key, ax

    #
    ### Iterators returning axes groups

    @property
    def axes_iter__row_axes(self):
        """Returns: row_lvl1, (ax11, ax21, ...) >> row_lvl2, (ax12, ax22, ...) >> ..."""
        for rowkey, axes in zip(self.levels_dim_dict["row"], self.axes_nested):
            yield rowkey, axes

    @property
    def axes_iter__col_axes(self):
        """Returns: col_lvl1, (ax11, ax21, ...) >> col_lvl2, (ax12, ax22, ...) >> ..."""
        for colkey, axes in zip(self.levels_dim_dict["col"], self.axes_nested.T):
            yield colkey, axes

    #
    ### Iterator yielding axes and DF

    @property
    def axes_iter__ax_df(self):
        """Returns: >> (ax11, df11) >> (ax12, df12) >> (ax21, df21) >> ..."""
        if self.factors_rowcol is None:
            yield self.axes, self.data  # * If no row or col, return all axes and data
        else:
            for (key_ax, ax), (key_df, df) in zip(
                self.axes_iter__keys_ax, self.data_iter__key_rowcol
            ):
                assert (
                    key_df == key_ax
                ), f"Mismatch of dataframe_key and ax_key: {key_df} != {key_ax}"
                # * Seaborn breaks on Dataframes that are only NaNs
                if df[self.dims.y].isnull().all():
                    continue
                else:
                    yield ax, df

    #
    ### Selective Iterators:

    @property
    def axes_iter_leftmost(self):
        """Returns: >> ax11 >> ax21 >> ax31 >> ax41 >> ..."""
        for row in self.axes_nested:
            yield row[0]

    @property
    def axes_iter_notleftmost(self):
        """Returns: >> axes excluding leftmost column"""
        for row in self.axes_nested:
            for ax in row[1:]:
                yield ax

    @property
    def axes_iter_lowerrow(self):
        """Returns: >> ax31 >> ax32 >> ax33 >> ax34 >> ..."""
        for ax in self.axes_nested[-1]:
            yield ax

    @property
    def axes_iter_notlowerrow(self):
        """Returns: >> axes excluding lowest row"""
        for row in self.axes_nested[:-1]:
            for ax in row:
                yield ax

    #
    #  ... PLOT ...........................................................................................

    def plot(
        self, kind: str = "strip", subplot_kws: dict = None, **sns_kws
    ) -> PlotTool:
        """Quick plotting initialising mpl.subplots and filling its axes with seaborn graphics

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

    def subplots(
        self,
        wspace=0.4,
        hspace=0.7,
        width_ratios: list[int] = None,
        height_ratios: list[int] = None,
        figsize: tuple[int] = None,
        **subplot_kws: dict,
    ) -> PlotTool:
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
        self.reset_axtitles()

        return self

    def subplots_snip(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self.DOCS['plt.subplots']} #\n"
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

    def fillaxes(self, kind: str = "strip", **sns_kws: dict) -> PlotTool:
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
            self.SNS_FUNCS[kind](
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

    def fillaxes_snip(self, kind: str = "strip", doclink=True) -> str:
        s = ""
        s += f"# . . . {self.DOCS[kind]} #\n"
        s += "kws = dict(alpha=.8) \n"
        s += "for ax, df in DA.iter_axes_and_data: \n"
        s += f"\t{self.SNS_FUNCS_STR[kind]}(data=df, ax=ax, y='{self.dims.y}', x='{self.dims.x}', hue='{self.dims.hue}', **kws)\n"
        s += "\tax.legend_.remove() \n"  # * Remove legend, unless we want one per axes"
        s += "DA.legend()  # * Add one standard legend to figure \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # ... Fig properties ...........................................................
    @property
    def figsize(self) -> tuple[int]:
        return self.fig.get_size_inches()

    #
    #
    #
    # ... EDIT Titles of fig & axes.........................................................................

    @staticmethod
    def _standard_axtitle(key: tuple[str] | str, connect="\n") -> str:
        """make axis title from key

        Args:
            key (tuple): _description_
        """
        if isinstance(key, str):
            return ut.capitalize(key)
        elif isinstance(key, tuple):
            key = [ut.capitalize(k) for k in key]
            return connect.join(key)

    def reset_axtitles(self) -> PlotTool:
        for key, ax in self.axes_iter__keys_ax:
            ax.set_title(self._standard_axtitle(key))
        return self

    def edit_titles(
        self,
        axes: mpl.axes.Axes = None,
        axtitles: dict = None,
    ) -> PlotTool:
        axes = axes or self.axes

        if not axtitles is None:
            for key, ax in self.axes_iter__keys_ax:
                ax.set_title(axtitles[key])
        return self

    def edit_format_titles(
        self,
        row_format: Callable = None,
        col_format: Callable = None,
        connect="\n",
    ) -> PlotTool:
        row_format = row_format or (lambda x: x)
        col_format = col_format or (lambda x: x)

        for rowkey, axes in self.axes_iter__row_axes:
            for ax in axes:
                title = row_format(rowkey)
                ax.set_title(title)
        for colkey, axes in self.axes_iter__col_axes:
            for ax in axes:
                title = ax.get_title() + connect + col_format(colkey)
                ax.set_title(title)
        return self

    def edit_format_titles_snip(self) -> str:
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

    def edit_replace_titles(self, titles: list) -> PlotTool:
        """Edits axes titles. If list is longer than axes, the remaining titles are ignored

        Args:
            titles (list): Titles to be set. The order of the titles should be the same as the order of the axes, which is from left to right for row after row (like reading).

        Returns:
            PlotTool: The object itselt
        """

        for ax, title in zip(self.axes_nested.flatten(), titles):
            ax.set_title(title)
        return self

    def edit_replace_titles_snip(self):
        s = ""
        s += f"titles = {[ax.get_title() for ax in self.axes.flatten()]} \n"
        s += "for ax, title in zip(DA.axes.flatten(), titles): \n"
        s += "\tax.set_title(title) \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    ### ... EDIT x- & y-axis LABELS ............................................................

    def edit_xyaxis_labels(
        self,
        leftmost: str = "",
        notleftmost: str = "",
        lowerrow: str = "",
        notlowerrow: str = "",
    ) -> PlotTool:
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

    def edit_xyaxis_labels_snip(self) -> str:
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
    ### ... EDIT x- & y-axis SCALE ............................................................

    def edit_log_yscale(
        self, base=10, nonpositive="clip", subs=[2, 3, 4, 5, 6, 7, 8, 9]
    ) -> PlotTool:
        for ax in self.axes_nested.flatten():
            ax.set_yscale(
                value="log",  # * "symlog", "linear", "logit", ...
                base=base,  # * Base of the logarithm
                nonpositive=nonpositive,  # * "mask": masked as invalid, "clip": clipped to a very small positive number
                subs=subs,  # * Where to place subticks between major ticks ! not working
            )

            # ax.yaxis.sety_ticks()
        return self

    def edit_log_xscale(
        self, base=10, nonpositive="clip", subs=[2, 3, 4, 5, 6, 7, 8, 9]
    ) -> PlotTool:
        for ax in self.axes.flatten():
            ax.set_xscale(
                value="log",  # * "symlog", "linear", "logit", ...
                base=base,  # * Base of the logarithm
                nonpositive=nonpositive,  # * "mask": masked as invalid, "clip": clipped to a very small positive number
                subs=subs,  # * Where to place subticks between major ticks ! not working
            )
        return self

    def edit_scale_snip(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self.DOCS['set_xscale']} #\n"
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
    # ... EDIT, Ticks and Ticklabels # ......................................................................

    def edit_yticklabel_percentage(
        self, decimals_major: int = 0, decimals_minor: int = 0
    ) -> PlotTool:
        for ax in self.axes_nested.flatten():
            ax.yaxis.set_major_formatter(
                mpl.ticker.PercentFormatter(xmax=1, decimals=decimals_major)
            )
            ax.yaxis.set_minor_formatter(
                mpl.ticker.PercentFormatter(xmax=1, decimals=decimals_minor)
            )
        return self

    def edit_yticklabel_percentage_snip(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self.DOCS['percent_formatter']} #\n"
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0)) \n"
        s += "\tax.yaxis.set_minor_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=1)) \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    def edit_add_minorticklabels(self, subs: list = [2, 3, 5, 7]):
        """minor ticklabels are kicked out unless their rounded mantissa (the digits from a float) is in subs

        Args:
            subs (list, optional): _description_. Defaults to [2, 3, 5, 7].

        Returns:
            _type_: _description_
        """
        for ax in self.axes_nested.flatten():
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

    def edit_add_minorticklabels_snip(self) -> str:
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

    def edit_xticklabels(
        self,
        lowerrow: list = None,
        notlowerrow: list = None,
        rotation: int = 0,
        ha: str = "center",
        va: str = "top",
        pad: float = 1,
        **text_kws,
    ) -> PlotTool:
        """Edits x- ticklabels

        Args:
            lowerrow (list): x-axis ticklabels for lower row of axes
            notlowerrow (list): x-axis ticklabels for not-lower row of axes
            etc.
        """
        notlowerrow = notlowerrow or self.levels_dim_dict["x"]
        lowerrow = lowerrow or self.levels_dim_dict["x"]

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

    def edit_xticklabels_snip(self) -> str:
        s = ""
        s += f"notlowerrow = {self.levels_dim_dict['x']} \n"
        s += f"lowerrow = {self.levels_dim_dict['x']} \n"
        s += "kws = dict( \n"
        s += "\trotation=0, #* Rotation in degrees \n"
        s += "\tha='center', #* Horizontal alignment [ 'center' | 'right' | 'left' ] \n"
        s += "\tva='top', #* Vertical Alignment   [ 'center' | 'top' | 'bottom' | 'baseline' ] \n"
        s += ") \n"
        s += f"ticks = {[i for i in range(len(self.levels_dim_dict['x']))]} \n"
        s += "for ax in DA.axes_iter_notlowerrow: \n"
        s += "\tax.set_xticks(ticks=ticks, labels=notlowerrow, **kws) \n"
        s += "\tax.tick_params(axis='x', pad=1) #* Sets distance to figure \n"
        s += "for ax in DA.axes_iter_lowerrow: \n"
        s += "\tax.set_xticks(ticks=ticks, labels=lowerrow, **kws) \n"
        s += "\tax.tick_params(axis='x', pad=1) #* Sets distance to figure \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    # ... EDIT Gridlines! # ......................................................................
    def edit_grid(self) -> PlotTool:
        for ax in self.axes_nested.flatten():
            ax.yaxis.grid(True, which="major", ls="-", linewidth=0.5, c="grey")
            ax.yaxis.grid(True, which="minor", ls="-", linewidth=0.2, c="grey")
            ax.xaxis.grid(True, which="major", ls="-", linewidth=0.3, c="grey")
        return self

    def edit_grid_snip(self) -> str:
        s = ""
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.yaxis.grid(True, which='major', ls='-', linewidth=0.5, c='grey') \n"
        s += "\tax.yaxis.grid(True, which='minor', ls='-', linewidth=0.2, c='grey') \n"
        s += "\tax.xaxis.grid(True, which='major', ls='-', linewidth=0.3, c='grey') \n"
        pyperclip.copy(s)
        print("#! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    # ... EDIT Shapes & Sizes
    # TODO: Until now, stick with the arguments supplied to self.subplots
    #
    #
    #
    # ... Add Legend ............................................................................

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

    def edit_legend(self) -> PlotTool:
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

    def edit_legend_snip(self, doclink=True) -> str:
        s = ""
        if doclink:
            s += f"# . . . {self.DOCS['legend']} #\n"
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

    # ... Fontsizes .....................................

    def edit_fontsizes(self, ticklabels=10, xylabels=10, axis_titles=10) -> PlotTool:
        """Edits fontsizes in [pt]. Does not affect legent or suptitle

        Args:
            ticklabels (int, optional): _description_. Defaults to 9.
            xylabels (int, optional): _description_. Defaults to 10.
            axis_titles (int, optional): _description_. Defaults to 11.

        Returns:
            PlotTool: _description_
        """

        for ax in PT.axes.flatten():
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

    def edit_fontsizes_snip(self) -> str:
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


# ! # end class
# !
# !
# !

# %% Initialize Data and PlotTool . . . . . . . . . . . . . . . . . . . . . . .

DF, dims = ut.load_dataset("tips")
DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
PT = PlotTool(data=DF, dims=DIMS).switch("x", "col")

# PT.data_describe()
# PT.plot()


# %% Experiments:
def tester(DF, dims):
    PT = PlotTool(data=DF, dims=dims)  # .switch("x", "col")

    PT: PlotTool = (
        PT.subplots(sharey=True)
        .fillaxes(kind="box", boxprops=dict(alpha=0.5))
        .fillaxes(kind="swarm", size=3, dodge=True)
        .reset_axtitles()
    )
    PT = (
        PT.edit_titles()
        .edit_format_titles()
        .edit_xyaxis_labels()
        .edit_log_yscale(base=10)
        .edit_yticklabel_percentage()
        .edit_add_minorticklabels(subs=[2, 3, 5, 7])
        .edit_xticklabels()
        .edit_grid()
        .edit_fontsizes(9, 10, 7)
        # .edit_replace_titles(titles = ["1", "2", "3", "4", "5", "6", "7", "8"])
    )
    if PT.dims.hue:  # * Only when legend
        PT = PT.edit_legend()
    # plt.close()


dimses = [
    # dict(y="tip", x="day", hue="sex", col="smoker", row="time"),
    # dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    # dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="sex", hue="day", col="time"),
    dict(y="tip", x="sex", hue="day", row="time"),
    dict(y="tip", x="sex", hue="day", row="size-cut"),
    dict(y="tip", x="sex", hue="day"),
    dict(y="tip", x="sex"),
]

DF, dims = ut.load_dataset("tips")
for dim in dimses:
    tester(DF, dim)

# %%

fig, axes = plt.subplots(1, 1)
print(axes)
axes = np.array([axes]).reshape(1, 1)
print(axes)
### make a 2D 1x1 array with 1 column and 1 row
# axes = np.array([axes]).reshape(1, 1)


# %% New Dataset . . . . . .
# DF, dims = ut.load_dataset("fmri")
# PT = PlotTool(data=DF, dims=dims)

# PT = PT.switch("row", "col").plot()
# PT.set(row="none", col="none").plot()
# PT = (
#     PT.switch("row", "col")
#     .subplots(sharey=True, height_ratios=[2, 1])
#     .fillaxes(kind="line", alpha=0.5)
#     .fillaxes(kind="strip", size=2, dodge=True, alpha=0.4)
#     # .edit_log_yscale(base=10)
#     .edit_grid()
#     .edit_legend()
#     # .edit_add_minorticklabels(subs=[2, 3, 5, 7])
#     .edit_fontsizes(9, 10, 11)
#     .edit_yticklabel_percentage(decimals_minor=1, decimals_major=1)
# )


# plt.close()


# %% main ................................................................................


def main():
    DF, dims = ut.load_dataset("tips")  # * Load Data. Dims
    DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
    PT = PlotTool(
        data=DF, dims=DIMS
    )  # ! We use PT, but please access these functions via the DataAnalysis class (DA)

    ### Test Parts
    PT.subplots()  # * Make Figure and Axes

    ### Test Interface functions
    PT = (
        PT.plot()
    )  # ! Make sure to return new instance of PlotTool, otherwise subsequent edits won't apply

    ### Plot in two steps
    PT.subplots()  # * Make empty axes
    PT.fillaxes(kind="swarm")  # * Fill axes with seaborn graphics
    plt.show()

    ### Plot in One step
    PT.plot(kind="bar")

    ### Access axes via PT.axes (of type np.ndarray[mpl.axes._subplots.AxesSubplot])
    for ax in PT.axes.flatten():
        ax.set_title("bla")

    ### Overlay two plots:
    PT.subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
    PT.fillaxes(kind="box", boxprops=dict(alpha=0.5))
    PT.fillaxes(kind="swarm", size=3, dodge=True)

    for i, ax in enumerate(PT.axes.flatten()):  # * Pick axes as you want!
        if i == 2:
            ax.set_title("THIRD!")

    ### Use Chaining
    # * (methods act inplace, so you have to re-initialize PT to start from scratch!)
    PT = (  # * Needs to be passed in order to make it modifyable in subequent lines
        PT.switch("x", "col")  # * Experiment with switching dimensions!
        .subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
        .fillaxes(kind="box", boxprops=dict(alpha=0.5))
        .fillaxes(kind="swarm", size=3, dodge=True)
        .edit_legend()  # * Add standard legend
    )

    for i, ax in enumerate(PT.axes.flatten()):
        if i == 2:
            ax.set_title("THIRD!")

    ### Don't memorize this, just copy code to the clipboard!
    PT.subplots_snip(doclink=True)
    DA = PT  # ! If you use DataAnalysis of PlotTool, it makes no difference!
    # !(We use DA, since its' not intended to use PT directly)

    ### There's a snippet for fillaxes too!
    PT.fillaxes_snip(kind="bar", doclink=True)
    # . . . https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot #
    # kws = dict(alpha=.8)
    # for ax, df in DA.iter_axes_and_data:
    #     sns.barplot(data=df, ax=ax, y='tip', x='day', hue='sex', **kws)
    #     ax.legend_.remove()
    # DA.edit_legend()  # * Add legend to figure

    ### A snippet for configuring Legends
    PT.edit_legend_snip()
    # # . . . https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.legend #
    # DA.fig.legend(
    #     title='sex', #* Hue factor
    #     handles=DA.legend_handles_and_labels[0],
    #     labels=DA.legend_handles_and_labels[1],
    #     loc='center right', #* Rough location
    #     bbox_to_anchor=(1.15, 0.50), #* Exact location in width, height relative to complete figure
    #     ncols=1, #* If >1, labels are displayed next to each other
    #     borderaxespad=3, #* Padding around axes, (pushing legend away)
    #     markerscale=1.5, #* Marker size relative to plotted datapoint
    #     frameon=False, #* Remove frame around legend
    # )

    # ! Snippets use matplotlib functions, which don't return PlotHelper object, so they can NOT be chained!
    # ! Use them at the end of a layer chain!

    ### Try Different Dataset
    DF, dims = ut.load_dataset("fmri")
    PT = PlotTool(data=DF, dims=dims)

    PT.plot()
    PT.switch("row", "col").plot()
    PT.set(row="none", col="none").plot()

    ### Logarithmic scale
    PT.edit_log_yscale(base=2)

    ### Snippet for Logarithmic scaling
    PT.edit_scale_snip(doclink=True)
    # # . . . https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale #
    # for ax in DA.axes.flatten():
    #     ax.set_yscale('log',  # * 'symlog', 'linear', 'logit',
    #         base=10,
    #         nonpositive='clip', # * 'mask': masked as invalid, 'clip': clipped to a very small positive number
    #     )
    #     # ax.set_xscale('log') # ? Rescale x-axis

    ### Gridlines
    PT.edit_grid()

    ### Snippet for Gridlines
    PT.edit_grid_snip()
    # for ax in PT.axes.flatten():
    #     ax.yaxis.grid(True, which='major', ls='-', linewidth=0.5, c='grey')
    #     ax.yaxis.grid(True, which='minor', ls='-', linewidth=0.2, c='grey')
    #     ax.xaxis.grid(True, which='major', ls='-', linewidth=0.3, c='grey')

    ### Show minor tick-labels
    PT.edit_add_minorticklabels(subs=[2, 3, 5, 7])

    ### Snippet for minor tick-labels
    PT.edit_add_minorticklabels_snip()
    # for ax in PT.axes.flatten():
    #     #* Set minor ticks, we need ScalarFormatter, others can't get casted into float
    #     ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter(useOffset=0, useMathText=False))
    #     #* Iterate through labels
    #     for label in ax.yaxis.get_ticklabels(which='minor'):
    #         # ? How else to cast float from mpl.text.Text ???
    #         label_f = float(str(label).split(', ')[1])  #* Cast to float
    #         mantissa = int(round(ut.mantissa_from_float(label_f))) #* Calculate mantissa
    #         if not mantissa in [2, 3, 5, 7]:
    #             label.set_visible(False) # * Set those not in subs to invisible

    ### Change Font Sizes
    PT.edit_fontsizes(ticklabels=9, xylabels=10, axis_titles=11)

    ### Snippet for Font Sizes
    PT.edit_fontsizes_snip()
    # ticklabels, xylabels, axis_titles = 9, 10, 11 ### <--- CHANGE THIS [pt]
    # for ax in PT.axes.flatten():
    #     ax.tick_params(axis='y', which='major', labelsize=ticklabels) # * Ticklabels
    #     ax.tick_params(axis='y', which='minor', labelsize=ticklabels-.5)
    #     ax.tick_params(axis='x', which='major', labelsize=ticklabels)
    #     ax.tick_params(axis='x', which='minor', labelsize=ticklabels-.5)
    #     ax.yaxis.get_label().set_fontsize(xylabels) # * xy-axis labels
    #     ax.xaxis.get_label().set_fontsize(xylabels)
    #     ax.title.set_fontsize(axis_titles) # * Title

    ### Format your axes titles:
    PT.edit_format_titles(
        row_format=lambda x: x.upper(),
        col_format=lambda x: "",
        connect=" || ",
    )

    ### Snippet for axes title formatting
    PT.edit_format_titles_snip()
    # row_format = lambda x: x.upper() #* e.g. try lambda x: x.upper()
    # col_format = lambda x: x
    # connect = '\n' #* newline. Try ' | ' as a separator in the same line
    # for rowkey, axes in PT.axes_iter__row_axes:
    #     for ax in axes:
    #         title = row_format(rowkey)
    #         ax.set_title(title)
    # for colkey, axes in PT.axes_iter__col_axes:
    #     for ax in axes:
    #         title = ax.get_title() + connect + col_format(colkey)
    #         ax.set_title(title)

    ### Change the axis titles fully manually
    PT.edit_replace_titles(
        [
            "guccy",
            "guccy-gu",
            "dada\nnana",
            "is this empty?",
            "next row",
            "...",
        ]
    )

    ### Snippet for replacing titles
    PT.edit_replace_titles_snip()
    # titles = ['Lunch \nThsdfr ', 'Lunch \nFri ', 'Lunch \nSat ', 'Lunch \nSun ', 'Dinner \nThur ', 'Dinner \nFri ', 'Dinner \nSat ', 'Dinner \nSun ']
    # for ax, title in zip(PT.axes.flatten(), titles):
    #     ax.set_title(title)

    ### Change the axis labels
    PT.edit_xyaxis_labels(
        leftmost="tipdf",
        notleftmost="",
        lowerrow="smoker",
        notlowerrow="d",
    )

    ### Snipper for changing the axis labels
    PT.edit_xyaxis_labels_snip()
    # ### y-axis labels
    # for ax in PT.axes_iter_leftmost:
    #     ax.set_ylabel('tip')
    # for ax in PT.axes_iter_notleftmost:
    #     ax.set_ylabel('')
    # ### x-axis labels
    # for ax in PT.axes_iter_lowerrow:
    #     ax.set_xlabel('smoker')
    # for ax in PT.axes_iter_notlowerrow:
    #     ax.set_xlabel('')

    ### Edit xtick-labels
    PT.edit_xticklabels(
        lowerrow=["NOO", "yup"],
        notlowerrow=["dd", "ee"],
        rotation=0,
        ha="center",
        pad=1,
    )
    ### Snippet for changing the axis labels
    PT.edit_xticklabels_snip()
    # DA=PT
    # notlowerrow = ['', '']
    # lowerrow = ['Yes', 'No']
    # kws = dict(
    #     rotation=30, #* Rotation in degrees
    #     ha='right', #* Horizontal alignment [ 'center' | 'right' | 'left' ]
    #     va='top', #* Vertical Alignment   [ 'center' | 'top' | 'bottom' | 'baseline' ]
    # )
    # ticks = [0, 1]
    # for ax in DA.axes_iter_notlowerrow:
    #     ax.set_xticks(ticks=ticks, labels=notlowerrow, **kws)
    #     ax.tick_params(axis='x', pad=.01) #* Sets distance to figure
    # for ax in DA.axes_iter_lowerrow:
    #     ax.set_xticks(ticks=ticks, labels=lowerrow, **kws)
    #     ax.tick_params(axis='x', pad=.01) #* Sets distance to figure


if __name__ == "__main__":
    # main()
    pass
