#
# %%
from cgitb import reset
from typing import Generator, TYPE_CHECKING

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

        # with ioff():
        self.fig, self.axes = plt.subplots(
            nrows=self.len_rowlevels, ncols=self.len_collevels
        )
        plt.close()  # * Close the figure to avoid displaying it

    # # ... Handle pyplot Hidden state
    # def suppress_display():
    #     plt.ioff()

    # ... Iterate through axes #...........................................................................................

    @property
    def iter_axes(self):
        """Associates axes with rowcol keys!
        Returns: {
            (R_lvl1, C_lvl1): ax11,
            (R_lvl1, C_lv2): ax12,
            (R_lvl2, C_lvl1): ax21,
            (R_lvl2, C_lvl2): ax22
        }"""
        for key, ax in zip(self.levelkeys_rowcol, self.axes.flatten()):
            yield key, ax

    @property
    def iter_axes_and_data(self):
        """Returns: ( (ax11, df11), (ax12, df12), (ax21, df21), ...)"""
        for (key_df, df), (key_ax, ax) in zip(self.iter_rowcol, self.iter_axes):
            assert key_df == key_ax, f"{key_df} != {key_ax}"
            yield ax, df

    # ... Plotting #...........................................................................................

    def subplots(
        self, **subplot_kws: dict
    ) -> tuple["mpl.figure.Figure", "mpl.axes.Axes"]:
        """Initialise matplotlib figure and axes objects

        Returns:
            tuple["mpl.figure.Figure", "mpl.axes.Axes"]: matplotlib figure and axes objects
        """

        # * https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
        self.fig, self.axes = plt.subplots(
            nrows=self.len_rowlevels,
            ncols=self.len_collevels,
            **subplot_kws,
        )
        ### Add titles to axes
        self.reset_axtitles()
        
        return self.fig, self.axes

    def fillaxes(
        self, kind: str = "strip", **sns_kws: dict
    ) -> mpl.axes.Axes:
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
        ), f"Axes size mismatch {axes.flatten().size} != {self.len_rowlevels * self.len_collevels}"

        ### Handle kwargs
        kws = dict(legend=False)
        kws.update(sns_kws)

        ### Iterate through data and axes
        for ax, df in self.iter_axes_and_data:
            sns.stripplot(
                data=df, ax=ax, y=self.dims.y, x=self.dims.x, hue=self.dims.hue, **kws
            )


    def plot(
        self, kind="strip", subplot_kws: dict = None, sns_kws: dict = None
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

        ### Initialise Figure and Axes
        self.subplots(**subplot_kws)
        
        ### Fill axes with seaborn graphics
        self.fillaxes(kind=kind, **sns_kws) 

        return self.fig, self.axes

    # ... Edit Titles .........................................................................

    @staticmethod
    def _standard_axtitle(key: tuple, connect=" | "):
        """make axis title from key

        Args:
            key (tuple): _description_
        """
        return connect.join(key)

    def reset_axtitles(self):
        for key, ax in self.iter_axes:
            ax.set_title(self._standard_axtitle(key))

    def edit_titles(
        self,
        axes: mpl.axes.Axes = None,
        axtitles: dict = None,
    ):
        axes = axes or self.axes

        if not axtitles is None:
            for key, ax in self.iter_axes:
                ax.set_title(axtitles[key])

    # ...

    # ... Show, Cache, Re-Use #...........................................................................................

    def show_plot(self):
        display(self.plot)


# ... PLAY AROUND .......................................................................................................

# %%
DF, dims = ut.load_dataset("tips")
DIMS = dict(y="tip", x="sex", hue="size-cut", col="smoker", row="time")
plottool = PlotTool(data=DF, dims=DIMS)

#%%
plottool.describe_data()

# %%
### Play around:
# plottool.axes
# for key, ax in plottool.iter_axes:
#     print(key, ax)

# fig, axes = plottool.plot()
# fig, axes = plottool.switch("col", "hue").plot()
# for (key, df), ax in zip(DF.groupby(["smoker", "time"]), axes.flatten()):
#     sns.stripplot(data=df, y="tip", x="sex", hue="size-cut", ax=ax, legend=False)
# plt.show()
# plottool.reset_axtitles()
fig, axes = plottool.plot()
# plottool.fillaxes(axes=axes)
plt.tight_layout()
plottool.fig
plt.show()

# %%

# %%
########... SUMMARIZE INTO TESTS .......................................................................................................


def main():
    DF, dims = ut.load_dataset("tips")  # * Load Data. Dims
    DIMS = dict(y="tip", x="sex", hue="size-cut", col="smoker", row="time")
    plottool = PlotTool(data=DF, dims=DIMS)

    ### Test Parts
    fig, axes = plottool.subplots()  # * Make Figure and Axes

    ### Test Interface functions
    plottool.plot()


if __name__ == "__main__":
    # main()
    pass
