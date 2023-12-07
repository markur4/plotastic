"""Plotting functions that aren't covered by matplotlib or seaborn."""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotastic.plotting.plotedits import PlotEdits
from plotastic.utils import utils as ut

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

# %%
# == Class Plot ========================================================


class Plot(PlotEdits):
    def __init__(self, **dataframetool_kws) -> None:
        super().__init__(**dataframetool_kws)

    def plot(
        self, kind: str = "strip", subplot_kws: dict = None, **sns_kws
    ) -> "Plot | DataAnalysis":
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

        self.subplots(**subplot_kws)  #' Initialise Figure and Axes
        self.fillaxes(kind=kind, **sns_kws)  #' Fill axes with seaborn graphics
        if self.dims.hue:
            self.edit_legend()  #' Add legend to figure

        plt.tight_layout()  #' Make sure everything fits nicely

        return self

    #
    # == Subject lines =================================================

    def _nested_offsets(self, n_levels, width=0.8, dodge=True) -> np.ndarray:
        """Return offsets for each hue level for dodged plots. This must
        represent the same function that seaborn uses to dodge the plot,
        which can be found here:
        https://github.com/mwaskom/seaborn/blob/908ca95137c0e73bb6ac9ce9a8051577b6453138/seaborn/categorical.py#L437
        """
        # ?? Retrieve hue_offsets from axes independently of width?
        # ?? This here could work, but might also be a bit hacky
        # axes = self.axes
        # offset = self.axes.collections[0].get_offsets()

        hue_offsets: np.ndarray
        if dodge:
            each_width = width / n_levels
            hue_offsets = np.linspace(0, width - each_width, n_levels)
            hue_offsets -= hue_offsets.mean()
        else:
            hue_offsets = np.zeros(n_levels)
        return hue_offsets

    @ut.ignore_warnings
    def _subjects_get_XY(self) -> pd.DataFrame:
        """Collects X and Y positions of all datapoints indexed by all
        factors and subjects in a dataframe"""
        if self.subject is None:
            raise TypeError("No subject column specified")

        ### Retrieve hue levels and relative x-positions of data on plot
        if self.dims.hue:
            all_hue_lvls = tuple(self.levels_dict_dim["hue"])
            hue_offset = self._nested_offsets(len(all_hue_lvls))

        get_y = lambda df: tuple(df[self.dims.y].tolist())

        XY_df = pd.DataFrame(
            index=self.data_hierarchicize(sort=False).index,
            columns=["X", "Y"],
            data=None,
        )
        if self.dims.hue is None:
            for key, df in self._iter__hlkey_df():
                XY_df.loc[key, "Y"] = get_y(df)
                XY_df.loc[key, "X"] = tuple(i for i in range(len(df)))
        else:
            for key, df in self._iter__hlkey_df():
                #' X_positions >> hue_positions
                # > [0, 1] >> [0.2, 1.2, 2.2] and [0.2, 1.2, 2.2]
                # > [0, 1] >> [0.2,      1.2] and [0.2, 1.2, 2.2]
                #' Get hue-indices of hue-levels that aren't missing
                hue_lvls = tuple(
                    df.index.get_level_values(self.dims.hue).unique()
                )
                hue_indices: list[int] = ut.index_of_matchelements(
                    i1=all_hue_lvls, i2=hue_lvls
                )

                ### Find out which x_index we are at
                if self.factors_is_unfacetted:
                    x_levels: list[str | int] = self.levels_dict_dim["x"]
                else:
                    x_levels = tuple(
                        XY_df.loc[key[:-1], :]
                        .index.get_level_values(self.dims.x)
                        .unique()
                    )
                x_level_index = x_levels.index(key[-1])

                ### Translate hue_indices into x_positions by adding offset
                hue_positions: tuple = tuple(
                    x_level_index + hue_offset[hue_indices]
                )

                XY_df.loc[key, "Y"] = get_y(df)
                XY_df.loc[key, "X"] = hue_positions

        return XY_df

    def plot_connect_subjects(self, **plot_kws) -> "Plot | DataAnalysis":
        """Joins subjects with lines. This is useful to see how subjects
        behave across x and hue levels. This is only possible if the
        subject column is specified.

        :raises TypeError: Requires subject column to be specified in
            DataAnalysis object
        :return: self
        :rtype: Plot | DataAnalysis
        """
        if self.subject is None:
            raise TypeError("No subject column specified")

        plot_KWS = dict(color="black", ls="-", zorder=2, alpha=0.3)
        plot_KWS.update(plot_kws)

        XY_df = self._subjects_get_XY()

        for key, df in XY_df.groupby(self._factors_hierarchical[:-1]):
            if self.factors_is_unfacetted:
                X, Y = df["X"], df["Y"]
                plt.plot(X, Y, **plot_KWS)

            else:
                for rowcolkey_ax, ax in self.axes_iter__keys_ax:
                    if self.factors_is_1_facet:
                        rowcolkey_xy = key[0]
                    else:
                        rowcolkey_xy = key[0:2]

                    if rowcolkey_ax == rowcolkey_xy:
                        X, Y = df["X"], df["Y"]
                        ax.plot(X, Y, **plot_KWS)
        return self
