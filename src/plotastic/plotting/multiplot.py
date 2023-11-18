#
# %% imports

from typing import TYPE_CHECKING

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import pyperclip

# import markurutils as ut
import plotastic.utils.utils as ut

from plotastic.plotting.plotedits import PlotEdits

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

# %% Matplotlib Runtime Config (RC)


# %% Class MultiPlot


class MultiPlot(PlotEdits):
    #
    # == Favorite kwargs ..........................................................................

    #
    # == __init__ .................................................................................

    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    def plot(
        self, kind: str = "strip", subplot_kws: dict = None, **sns_kws
    ) -> "MultiPlot | DataAnalysis":
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
    # == Boxplots .................................................................................

    def plot_box_strip(
        self,
        marker_size: float = 2,
        marker_alpha: float = 0.5,
        legend=True,
        subplot_kws: dict = dict(),
        box_kws: dict = dict(),
        strip_kws: dict = dict(),
    ) -> "MultiPlot | DataAnalysis":
        """A boxplot with a stripplott (scatter) on top

        Args:
            markersize (float, optional): _description_. Defaults to 2.
            markeralpha (float, optional): _description_. Defaults to 0.5.
            box_kws (dict, optional): _description_. Defaults to dict().
            strip_kws (dict, optional): _description_. Defaults to dict().
        """
        # == PARAMETERS
        ### Linewidths
        thin, thick = 0.3, 1.0
        ### Alpha
        covering, translucent, hazy = 1.0, 0.5, 0.3
        ### z-order
        front, mid, background, hidden = 100, 50, 1, -1

        ### == KEYWORD ARGUMENTS
        ### Boxplot kws
        box_KWS = dict(
            showfliers=False,
            boxprops=dict(  # * Box line and surface
                alpha=hazy,
                linewidth=thin,
            ),
            medianprops=dict(  # * Median line
                alpha=covering,
                zorder=front,
                linewidth=thick,
            ),
            whiskerprops=dict(  # * Lines conencting box and caps
                alpha=covering,
                zorder=mid,
                linewidth=thin,
            ),
            capprops=dict(  # * Caps at the end of whiskers
                alpha=covering,
                zorder=mid,
                linewidth=thick,
            ),
        )

        ### Stripplot kws
        strip_KWS = dict(
            dodge=True,  # * Separates the points in hue
            jitter=0.2,  # * How far datapoints of one group scatter across the x-axis
            zorder=front,
            ### Marker Style
            alpha=marker_alpha,
            size=marker_size,
            # color="none",
            edgecolor="white",
            linewidth=thin,  # * Edge width of the marker
        )

        ### User KWS
        box_KWS.update(box_kws)
        strip_KWS.update(strip_kws)

        ###... PLOT
        (
            self.subplots(**subplot_kws)
            .fillaxes(kind="box", **box_KWS)
            .fillaxes(kind="strip", **strip_KWS)
        )

        if legend:
            self.edit_legend()

        return self


    def plot_box_swarm(
        self,
        marker_size: float = 1.5,
        marker_alpha: float = 0.9,
        legend=True,
        subplot_kws: dict = dict(),
        box_kws: dict = dict(),
        swarm_kws: dict = dict(),
    ) -> "MultiPlot | DataAnalysis":
        """A boxplot with a stripplott (scatter) on top

        Args:
            markersize (float, optional): _description_. Defaults to 2.
            markeralpha (float, optional): _description_. Defaults to 0.5.
            box_kws (dict, optional): _description_. Defaults to dict().
            strip_kws (dict, optional): _description_. Defaults to dict().
        """
        # == PARAMETERS
        ### Linewidths
        thin, thick = 0.2, 1.0
        ### Alpha
        covering, translucent, hazy = 1.0, 0.5, 0.3
        ### z-order
        front, mid, background, hidden = 100, 50, 1, -1

        ### == KEYWORD ARGUMENTS
        ### Boxplot kws
        box_KWS = dict(
            showfliers=False,
            # * Widths of boxes
            # !! Throws TypeError: matplotlib.axes._axes.Axes.boxplot() got multiple values for keyword argument 'widths'
            # widths=0.9,
            boxprops=dict(  # * Box line and surface
                alpha=translucent,
                linewidth=thin,
            ),
            medianprops=dict(  # * Median line
                alpha=covering,
                zorder=front,
                linewidth=thick,
            ),
            whiskerprops=dict(  # * Lines conencting box and caps
                alpha=covering,
                zorder=mid,
                linewidth=thin,
            ),
            capprops=dict(  # * Caps at the end of whiskers
                alpha=covering,
                zorder=mid,
                linewidth=thick,
            ),
        )

        ### Swarmplot kws

        swarm_KWS = dict(
            dodge=True,  # * Separates the points in hue
            zorder=front,
            ### Marker Style
            alpha=marker_alpha,
            size=marker_size,
            # color="none",
            edgecolor="black",
            linewidth=thin,  # * Edge width of the marker
        )

        ### User KWS
        box_KWS.update(box_kws)
        swarm_KWS.update(swarm_kws)

        ###... PLOT
        # !! If log y scale, you should pass y_scale = "log" in sublot_kws! Otherwise Points will not cluster in the middle!
        (
            self.subplots(**subplot_kws)
            .fillaxes(kind="box", **box_KWS)
            .fillaxes(kind="swarm", **swarm_KWS)
        )

        if legend:
            self.edit_legend()

        return self


## !!__________________________________________________________________________

# # %% Matplotlib Runtime Config (RC)

# mpl.rc("figure", dpi=250)

# # %% get data

# MP = MultiPlot(data=df, dims=dims)


# # %%
