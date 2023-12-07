#
# %% imports

from typing import TYPE_CHECKING

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

# import pyperclip

# import markurutils as ut
import plotastic.utils.utils as ut

# from plotastic.plotting.plotedits import PlotEdits
from plotastic.plotting.plot import Plot

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

# %% Matplotlib Runtime Config (RC)


# %% Class MultiPlot


class MultiPlot(Plot):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    #
    # == Boxplots ======================================================

    def plot_box_strip(
        self,
        marker_size: float = 2,
        marker_alpha: float = 0.5,
        legend=True,
        subplot_kws: dict = dict(),
        box_kws: dict = dict(),
        strip_kws: dict = dict(),
        legend_kws: dict = dict(),
    ) -> "MultiPlot | DataAnalysis":
        """A boxplot with a stripplott (scatter) on top

        Args:
            markersize (float, optional): _description_. Defaults to 2.
            markeralpha (float, optional): _description_. Defaults to 0.5.
            box_kws (dict, optional): _description_. Defaults to dict().
            strip_kws (dict, optional): _description_. Defaults to dict().
        """
        # == PARAMETERS
        thin, thick = 0.3, 1.0  #' Linewidths
        covering, translucent, hazy = 1.0, 0.5, 0.3  #' Alpha
        front, mid, background, hidden = 100, 50, 1, -1  #' z-order

        ### == KEYWORD ARGUMENTS
        ### Boxplot kws
        box_KWS = dict(
            showfliers=False,
            boxprops=dict(  #' Box line and surface
                alpha=hazy,
                linewidth=thin,
            ),
            medianprops=dict(  #' Median line
                alpha=covering,
                zorder=front,
                linewidth=thick,
            ),
            whiskerprops=dict(  #' Lines conencting box and caps
                alpha=covering,
                zorder=mid,
                linewidth=thin,
            ),
            capprops=dict(  #' Caps at the end of whiskers
                alpha=covering,
                zorder=mid,
                linewidth=thick,
            ),
        )

        ### Stripplot kws
        strip_KWS = dict(
            dodge=True,  #' Separates the points in hue
            jitter=0.2,  #' How far datapoints of one group scatter across the x-axis
            zorder=front,
            ### Marker Style
            size=marker_size,
            alpha=marker_alpha,
            # color="none",
            edgecolor="white",
            linewidth=thin,  #' Edge width of the marker
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

        ### Legend displaying labels of stripplot (since that was called last)
        if legend and self.dims.hue:
            self.edit_legend(**legend_kws)

        return self

    def plot_box_swarm(
        self,
        marker_size: float = 1.5,
        marker_alpha: float = 0.9,
        legend=True,
        subplot_kws: dict = dict(),
        box_kws: dict = dict(),
        swarm_kws: dict = dict(),
        legend_kws: dict = dict(),
    ) -> "MultiPlot | DataAnalysis":
        """A boxplot with a stripplott (scatter) on top

        Args:
            markersize (float, optional): _description_. Defaults to 2.
            markeralpha (float, optional): _description_. Defaults to 0.5.
            box_kws (dict, optional): _description_. Defaults to dict().
            strip_kws (dict, optional): _description_. Defaults to dict().
        """
        # == PARAMETERS
        thin, thick = 0.2, 1.0  #' Linewidths
        covering, translucent, hazy = 1.0, 0.5, 0.3  #' Alpha
        front, mid, background, hidden = 100, 50, 1, -1  #' z-order

        ### == KEYWORD ARGUMENTS
        ### Boxplot kws
        box_KWS = dict(
            showfliers=False,
            #' Widths of boxes
            # !! Throws TypeError: matplotlib.axes._axes.Axes.boxplot() got multiple values for keyword argument 'widths'
            # widths=0.9,
            boxprops=dict(  #' Box line and surface
                alpha=translucent,
                linewidth=thin,
            ),
            medianprops=dict(  #' Median line
                alpha=covering,
                zorder=front,
                linewidth=thick,
            ),
            whiskerprops=dict(  #' Lines conencting box and caps
                alpha=covering,
                zorder=mid,
                linewidth=thin,
            ),
            capprops=dict(  #' Caps at the end of whiskers
                alpha=covering,
                zorder=mid,
                linewidth=thick,
            ),
        )

        ### Swarmplot kws
        swarm_KWS = dict(
            dodge=True,  #' Separates the points in hue
            zorder=front,
            ### Marker Style
            alpha=marker_alpha,
            size=marker_size,
            # color="none",
            edgecolor="black",
            linewidth=thin,  #' Edge width of the marker
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

        ### Legend displaying labels of swarmplot (since that was called last)
        if legend and self.dims.hue:
            self.edit_legend(**legend_kws)

        return self


## !!__________________________________________________________________________

# # %% Matplotlib Runtime Config (RC)

# mpl.rc("figure", dpi=250)

# # %% get data

# MP = MultiPlot(data=df, dims=dims)


# # %%
