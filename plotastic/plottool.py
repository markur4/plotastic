#
# %%
from cgitb import reset
from re import T
from turtle import width
from typing import Generator, TYPE_CHECKING
from click import edit
from matplotlib import axes

import pyperclip

# from matplotlib import axes
import markurutils as ut
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from analysis import Analysis

if TYPE_CHECKING:
    import numpy as np


# %%
# !
# !
# !


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
        "plt.subplots": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots",
        "Axes.set_xscale": "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale",
        "legend": "https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.legend",
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

    #
    #
    # ...__INIT__ .....................

    def __init__(self, data: pd.DataFrame, dims: dict, verbose=False) -> "PlotTool":
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
        self.axes: np.ndarray[mpl.axes._subplots.AxesSubplot] = axes
        plt.close()  # * Close the figure to avoid displaying it

    #
    #
    #
    # ... ITERATORS #...........................................................................................

    @property
    def iter_keys_and_axes(self):
        """Returns: (R_lvl1, C_lvl1), ax11 / (R_lvl1, C_lv2), ax12 / (R_lvl2, C_lvl1), ax21 / ..."""
        for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
            yield key, ax

    @property
    def iter_axes_and_data(self):
        """Returns:  (ax11, df11) / (ax12, df12) / (ax21, df21) / ..."""
        for (key_ax, ax), (key_df, df) in zip(
            self.iter_keys_and_axes, self.iter_rowcol
        ):
            assert key_df == key_ax, f"{key_df} != {key_ax}"
            # * Seaborn breaks on Dataframes that are only NaNs
            if df[self.dims.y].isnull().all():
                continue
            else:
                yield ax, df

    #
    # ... PLOT ...........................................................................................

    def plot(
        self, kind: str = "strip", subplot_kws: dict = None, **sns_kws
    ) -> "PlotTool":
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

        self.subplots(**subplot_kws)  # * Initialise Figure and Axes
        self.fillaxes(kind=kind, **sns_kws)  # * Fill axes with seaborn graphics
        self.legend()  # * Add legend to figure
        plt.tight_layout()  # * Make sure everything fits nicely

        return self

    def subplots(self, **subplot_kws: dict) -> "PlotTool":
        """Initialise matplotlib figure and axes objects

        Returns:
            tuple["mpl.figure.Figure", "mpl.axes.Axes"]: matplotlib figure and axes objects
        """
        self.fig, self.axes = plt.subplots(
            nrows=self.len_rowlevels,
            ncols=self.len_collevels,
            **subplot_kws,
        )
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
        print(" ! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    def fillaxes(self, kind: str = "strip", **sns_kws: dict) -> "PlotTool":
        """_summary_

        Args:
            axes (mpl.axes.Axes): _description_
            kind (str, optional): _description_. Defaults to "strip".

        Returns:
            mpl.axes.Axes: _description_
        """
        ### Make sure axes is the same size as the number of row multiplied by column levels
        assert (
            self.axes.flatten().size == self.len_rowlevels * self.len_collevels
        ), f"Axes size mismatch {self.axes.flatten().size} != {self.len_rowlevels * self.len_collevels}"

        ### Handle kwargs
        kws = dict()
        kws.update(sns_kws)
        # print(dict(y=self.dims.y, x=self.dims.x, hue=self.dims.hue))

        ### Iterate through data and axes
        for ax, df in self.iter_axes_and_data:
            self.SNS_FUNCS[kind](
                data=df,
                ax=ax,
                y=self.dims.y,
                x=self.dims.x,
                hue=self.dims.hue,
                **kws,
            )

            # * Remove legend per axes, since we want one legend for the whole figure
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
        print(" ! Code copied to clipboard, press Ctrl+V to paste:")
        return s

    #
    # ... EDIT TITLES OF AXES.........................................................................

    @staticmethod
    def _standard_axtitle(key: tuple, connect=" | ") -> str:
        """make axis title from key

        Args:
            key (tuple): _description_
        """
        return connect.join(key)

    def reset_axtitles(self):
        for key, ax in self.iter_keys_and_axes:
            ax.set_title(self._standard_axtitle(key))

    def edit_titles(
        self,
        axes: mpl.axes.Axes = None,
        axtitles: dict = None,
    ):
        axes = axes or self.axes

        if not axtitles is None:
            for key, ax in self.iter_keys_and_axes:
                ax.set_title(axtitles[key])

    #
    ### ... EDIT X- & Y-axis LABELS ............................................................

    #
    ### ... EDIT X- & Y-axis SCALE ............................................................

    #
    ### ...

    #
    # ... EDIT, Ticks and Ticklabels # ......................................................................

    def edit_logscale(self, base=10, **scale_kws):
        for ax in self.axes.flatten():
            ax.set_xscale("log")  # * "symlog", "linear", "logit", ...
            ax.set_yscale("log")

    def edit_scale_snip(self):
        s = ""
        s += "for ax in DA.axes.flatten(): \n"
        s += "\tax.set_xscale('log') # * 'symlog', 'linear', 'logit', ... \n"
        s += "\tax.set_yscale('log') \n"
        return s

    def edit_ticks(self, **tick_kws):
        for ax in self.axes.flatten():
            ax.tick_params(**tick_kws)

    #
    # ... LEGEND ............................................................................

    @property
    def legend_handles_and_labels(self):
        handles, labels = self.axes.flatten()[0].get_legend_handles_labels()
        ### Remove duplicate handles from repeated plot layers
        by_label = dict(zip(labels, handles))
        handles = by_label.values()
        labels = by_label.keys()
        return handles, labels

    def legend(self) -> "PlotTool":
        """Adds standard legend to figure"""
        self.fig.legend(
            title=self.dims.hue.capitalize(),
            handles=self.legend_handles_and_labels[0],
            labels=self.legend_handles_and_labels[1],
            loc="center right",
            borderaxespad=0.1,
            bbox_to_anchor=(1.15, 0.5),
            frameon=False,
        )
        return self

    def legend_snip(self, doclink=True):
        s = ""
        if doclink:
            s += f"# . . . {self.DOCS['legend']} #\n"
        s += "DA.fig.legend( \n"
        s += f"\ttitle='{self.dims.hue}', #* Hue factor \n"
        s += "\thandles=DA.legend_handles_and_labels[0], \n"
        s += "\tlabels=DA.legend_handles_and_labels[1], \n"
        s += "\tloc='center right', #* Rough location \n"
        s += "\tbbox_to_anchor=(1.15, 0.50), #* Exact location in width, height relative to complete figure \n"
        s += "\tncols=1, #* If >1, labels are displayed next to each other \n"
        s += "\tborderaxespad=3, #* Padding around axes, (pushing legend away) \n"
        s += "\tmarkerscale=1.5, #* Marker size relative to plotted datapoint \n"
        s += "\tframeon=False, #* Remove frame around legend \n"
        s += ")\n"
        pyperclip.copy(s)
        print(" ! Code copied to clipboard, press Ctrl+V to paste:")
        return s


# !
# !
# !
# !
# !


# %%
# ... PLAY AROUND .......................................................................................................

DF, dims = ut.load_dataset("tips")
DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
PT = PlotTool(data=DF, dims=DIMS)

# %%
PT.plot()
PT.describe_data()


# %%
### Use Chaining
PT = (
    PT.switch("x", "col")
    .subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
    .fillaxes(kind="box", boxprops=dict(alpha=0.5))
    .fillaxes(kind="swarm", size=3, dodge=True)
    # .legend()  # * Add legend
)
for i, ax in enumerate(PT.axes.flatten()):
    if i == 2:
        ax.set_title("THIRD!")

PT.legend_snip()
DA = PT
# . . . https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.legend #
DA.fig.legend(
    title="sex",  # * Hue factor
    handles=DA.legend_handles_and_labels[0],
    labels=DA.legend_handles_and_labels[1],
    loc="center right",  # * Rough location
    bbox_to_anchor=(
        1.15,
        0.50,
    ),  # * Exact location in width, height relative to complete figure
    ncols=1,  # * If >1, labels are displayed next to each other
    borderaxespad=3,  # * Padding around axes, (pushing legend away)
    markerscale=1.5,  # * Marker size relative to plotted datapoint
    frameon=False,  # * Remove frame around legend
)


# %%
# Summarize
#
#
########... SUMMARIZE .......................................................................................................


def main():
    DF, dims = ut.load_dataset("tips")  # * Load Data. Dims
    DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
    PT = PlotTool(
        data=DF, dims=DIMS
    )  # ! We use PT, but please access these functions via the DataAnalysis class (DA)

    ### Test Parts
    PT.subplots()  # * Make Figure and Axes

    ### Test Interface functions
    PT.plot()

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
        .legend()  # * Add standard legend
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
    # DA.legend()  # * Add legend to figure

    ### A snippet for configuring Legends
    PT.legend_snip()
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


if __name__ == "__main__":
    # main()
    pass

# %%
