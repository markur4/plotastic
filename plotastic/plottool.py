#
# %%
from cgitb import reset
from turtle import width
from typing import Generator, TYPE_CHECKING

import pyperclip

# from matplotlib import axes
import markurutils as ut
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from analysis import Analysis

### Signatures
# * https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
# * https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes
fig_and_axes = (
    tuple[mpl.figure.Figure, mpl.axes.Axes]
    | list[mpl.figure.Figure, mpl.axes.Axes]
    | None
)


# ... context manager  for controlling display of figures...........................


# class ioff:
#     def __init__(self):
#         self._state = plt.isinteractive()
#         plt.ioff()

#     # def __call__(self):
#     #     mpl.interactive(False)
#     #     uninstall_repl_displayhook()

#     def __enter__(self):
#         # self.call()
#         return self

#     def __exit__(self, *args):
#         if self._state:
#             plt.ion()

# class ion:
#     def __init__(self):
#         self._state = plt.isinteractive()
#         plt.ion()

#     def __enter__(self):
#         return self

#     def __exit__(self, *args):
#         if not self._state:
#             plt.ioff()


# ... PlotTool ...........................


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

    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        verbose=False,
        # fig: mpl.figure.Figure = None,
        # axes: mpl.axes.Axes = None,
    ) -> "PlotTool":
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
        super().__init__(data=data, dims=dims, verbose=verbose)

        ### Initialise figure and axes
        self.fig, self.axes = plt.subplots(
            nrows=self.len_rowlevels, ncols=self.len_collevels
        )
        plt.close()  # * Close the figure to avoid displaying it

    # ... ITERATORS #...........................................................................................

    @property
    def iter_keys_and_axes(self):
        """Returns: (R_lvl1, C_lvl1), ax11 / (R_lvl1, C_lv2), ax12 / (R_lvl2, C_lvl1), ax21 / ..."""
        for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
            yield key, ax

    @property
    def iter_keys_and_axes_and_data(self):
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

    # ... PLOT ...........................................................................................

    def plot(
        self, kind: str = "strip", subplot_kws: dict = None, **sns_kws
    ) -> fig_and_axes:
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

    def subplots(
        self, **subplot_kws: dict
    ) -> tuple["mpl.figure.Figure", "mpl.axes.Axes"]:
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

    def subplots_snip(self, doclink=True):
        s = ""
        if doclink:
            s += f"# . . . {self.DOCS['plt.subplots']} #\n"
        s += "DA = DataAnalysis(data=DF, dims=DIMS)"
        s += f"""
        DA.fig, DA.axes = plt.subplots(
            nrows=DA.len_rowlevels, 
            ncols=DA.len_collevels,
            sharex=False, sharey=False, 
            width_ratios={[1 for _ in range(self.len_collevels)]}, height_ratios={[1 for _ in range(self.len_rowlevels)]},
            gridspec_kw=dict(wspace=0.2, hspace=0.5),
            subplot_kw=None, 
            )\n"""
        s += "DA.reset_axtitles()  # * Add basic titles to axes"
        s = s.replace("        ", "")
        pyperclip.copy(s)
        print(" ! This was copied to clipboard, press Ctrl+V to paste:")
        print(s)

    def fillaxes(self, kind: str = "strip", **sns_kws: dict) -> mpl.axes.Axes:
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
        for ax, df in self.iter_keys_and_axes_and_data:
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

    def fillaxes_snip(self, kind: str = "strip", doclink=True):
        s = ""
        s += f"# . . . {self.DOCS[kind]} #\n"
        s += "kws = dict(alpha=.8) \n"
        s += "for ax, df in DA.iter_keys_and_axes_and_data: \n"
        s += f"\t{self.SNS_FUNCS_STR[kind]}(data=df, ax=ax, y='{self.dims.y}', x='{self.dims.x}', hue='{self.dims.hue}', **kws)\n"
        s += "\tax.legend_.remove() \n"  # * Remove legend from axes"""
        s += "DA.legend()  # * Add legend to figure \n"
        pyperclip.copy(s)
        print(" ! This was copied to clipboard, press Ctrl+V to paste:")
        print(s)

    # ... EDIT TITLES .........................................................................

    @staticmethod
    def _standard_axtitle(key: tuple, connect=" | "):
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

    # ... Legend............................................................................

    @property
    def legend_handles_and_labels(self):
        handles, labels = self.axes.flatten()[0].get_legend_handles_labels()
        ### Remove duplicate handles from repeated plot layers
        by_label = dict(zip(labels, handles))
        handles = by_label.values()
        labels = by_label.keys()
        return handles, labels

    def legend(self, handles=None, labels=None, **legend_kws):
        """Add legend to figure"""
        self.fig.legend(
            title=self.dims.hue,
            handles=self.legend_handles_and_labels[0],
            labels=self.legend_handles_and_labels[1],
            loc="center right",
            bbox_to_anchor=(1.15, 0.5),
        )
        return self

    def legend_snip(self):
        raise NotImplementedError

    # ... Ticks and Ticklabels # ......................................................................

    def edit_ticks(self, **tick_kws):
        for ax in self.axes.flatten():
            ax.tick_params(**tick_kws)

    # ... Show, Cache, Re-Use #...........................................................................................

    def show_plot(self):
        display(self.plot)


# ... PLAY AROUND .......................................................................................................

# %%
DF, dims = ut.load_dataset("tips")
DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
PT = PlotTool(data=DF, dims=DIMS)

# %%
PT.plot()
PT.describe_data()


# %%
### Use Chaining
(
    PT.switch("x", "col")
    .subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
    .fillaxes(kind="box", boxprops=dict(alpha=0.5))
    .fillaxes(kind="swarm", size=3, dodge=True)
    .legend()  # * Add legend
)
for i, ax in enumerate(PT.axes.flatten()):
    if i == 2:
        ax.set_title("THIRD!")


# %%

### Don't memorize this, just copy code to the clipboard!
PT.fillaxes_snip(kind="bar", doclink=True)
DA = PT
PT.subplots()
# . . . https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot #
kws = dict(alpha=0.8)
for ax, df in DA.iter_keys_and_axes_and_data:
    sns.barplot(data=df, ax=ax, y="tip", x="day", hue="sex", **kws)
    ax.legend_.remove()
DA.legend()  # * Add legend to figure


# %%
########... SUMMARIZE INTO TESTS .......................................................................................................


def main():
    DF, dims = ut.load_dataset("tips")  # * Load Data. Dims
    DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
    PT = PlotTool(data=DF, dims=DIMS)

    ### Test Parts
    fig, axes = PT.subplots()  # * Make Figure and Axes

    ### Test Interface functions
    PT.plot()

    ### Plot in two steps
    PT.subplots()  # * Make empty axes
    PT.fillaxes(kind="swarm")  # * Fill axes with seaborn graphics
    plt.show()

    ### Plot in One step
    PT.plot(kind="bar")

    ### Access Axes
    for ax in PT.axes.flatten():
        ax.set_title("bla")

    ### Overlay two plots:
    PT.subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
    PT.fillaxes(kind="box", boxprops=dict(alpha=0.5))
    PT.fillaxes(kind="swarm", size=3, dodge=True)

    for i, ax in enumerate(axes.flatten()):  # * Pick axes as you want!
        if i == 2:
            ax.set_title("THIRD!")

    ### Use Chaining
    # * (methods act inplace, so you have to re-initialize PT to start from scratch!)
    (
        PT.switch("x", "col")  # * Experiment with switching dimensions!
        .subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
        .fillaxes(kind="box", boxprops=dict(alpha=0.5))
        .fillaxes(kind="swarm", size=3, dodge=True)
        .legend()  # * Add legend
    )

    for i, ax in enumerate(PT.axes.flatten()):
        if i == 2:
            ax.set_title("THIRD!")

    ### Don't memorize this, just copy code to the clipboard!
    PT.subplots_snip(doclink=True)


if __name__ == "__main__":
    # main()
    pass

# %%
