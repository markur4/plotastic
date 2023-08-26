#
# %% imports

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import markurutils as ut
from pyparsing import line
from plotastic.plottool import PlotTool


# %% Matplotlib Runtime Config (RC)


# %% Class MultiPlot


class MultiPlot(PlotTool):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    def mp_box_strip(
        self,
        markersize: float = 2,
        box_kws: dict = dict(),
        strip_kws: dict = dict(),
    ):
        """A boxplot with a stripplott (scatter) on top

        Args:
            markersize (float, optional): _description_. Defaults to 2.
            box_kws (dict, optional): _description_. Defaults to dict().
            strip_kws (dict, optional): _description_. Defaults to dict().
        """
        # ... PARAMETERS
        ### Linewidths
        thin = 0.3
        thick = 1.0
        ### Alpha
        covering = 1.0
        translucent = 0.5
        hazy = 0.3
        ### z-order
        front = 100
        mid = 50
        background = 1
        hidden = -1

        ### ... KEYWORD ARGUMENTS
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
        box_KWS.update(box_kws)

        ### Stripplot kws
        strip_KWS = dict(
            dodge=True,  # * Separates the points in hue
            jitter=0.2,  # * How far the points scatter across the x-axis
            zorder=front,
            ### Marker Style
            alpha=translucent,
            size=markersize,
            # color="none",
            edgecolor="white",
            linewidth=thin,  # * Edge width of the marker
            # facecolors='none',
        )
        strip_KWS.update(strip_kws)

        ###... PLOT
        (
            self.subplots()
            .fillaxes(kind="box", **box_KWS)
            .fillaxes(kind="strip", **strip_KWS)
        )

        return self
    
    def mp_box_strip_SNIP(self):
        pass


# !
# !  __________________________________________________________________________

## %% Matplotlib Runtime Config (RC)
mpl.rc("figure", dpi=250)

# %% get data

df, dims = ut.load_dataset("tips")
# df, dims = ut.load_dataset("fmri")

DA = MultiPlot(data=df, dims=dims)

DA = DA.mp_box_strip()

# %%
