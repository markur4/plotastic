#
# %% imports

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import pyperclip

import markurutils as ut

from plotastic.plotting.plottool import PlotTool


# %% Matplotlib Runtime Config (RC)


# %% Class MultiPlot


class MultiPlot(PlotTool):
    #
    # ... Favorite kwargs ..........................................................................

    #
    # ... __init__ .................................................................................
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    #
    # ... Boxplots .................................................................................

    def plot_box_strip(
        self,
        marker_size: float = 2,
        marker_alpha: float = 0.5,
        legend=True,
        box_kws: dict = dict(),
        strip_kws: dict = dict(),
    ):
        """A boxplot with a stripplott (scatter) on top

        Args:
            markersize (float, optional): _description_. Defaults to 2.
            markeralpha (float, optional): _description_. Defaults to 0.5.
            box_kws (dict, optional): _description_. Defaults to dict().
            strip_kws (dict, optional): _description_. Defaults to dict().
        """
        # ... PARAMETERS
        ### Linewidths
        thin, thick = 0.3, 1.0
        ### Alpha
        covering, translucent, hazy = 1.0, 0.5, 0.3
        ### z-order
        front, mid, background, hidden = 100, 50, 1, -1

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
            jitter=0.2,  # * How far datapoints of one group scatter across the x-axis
            zorder=front,
            ### Marker Style
            alpha=marker_alpha,
            size=marker_size,
            # color="none",
            edgecolor="white",
            linewidth=thin,  # * Edge width of the marker
        )
        strip_KWS.update(strip_kws)

        ###... PLOT
        (
            self.subplots()
            .fillaxes(kind="box", **box_KWS)
            .fillaxes(kind="strip", **strip_KWS)
        )

        if legend:
            self.edit_legend()

        return self

    def plot_box_strip_SNIP(self, doclink=True):
        s = "\n"
        if doclink:
            s += f"# {'Docs Boxplot:'.ljust(20)} {self.DOCS['box']}\n"
            s += f"# {'Docs Stripplot:'.ljust(20)} {self.DOCS['strip']}\n"
        s += "\n"
        s += "### ... PARAMETERS\n"
        s += "### Linewidths\n"
        s += "thin, thick = 0.3, 1.0\n"
        s += "### Alpha\n"
        s += "covering, translucent, hazy = 1.0, .5, .3\n"
        s += "### z-order\n"
        s += "front, mid, background, hidden = 100, 50, 1, -1\n"
        s += "\n"
        s += "### ... KEYWORD ARGUMENTS\n"
        s += "### Boxplot kws\n"
        s += "box_KWS = dict(\n"
        s += "    showfliers=False,\n"
        s += "    boxprops=dict(  # * Box line and surface\n"
        s += "        alpha=hazy,\n"
        s += "        linewidth=thin,\n"
        s += "    ),\n"
        s += "    medianprops=dict(  # * Median line\n"
        s += "        alpha=covering,\n"
        s += "        zorder=front,\n"
        s += "        linewidth=thick,\n"
        s += "    ),\n"
        s += "    whiskerprops=dict(  # * Lines conencting box and caps\n"
        s += "        alpha=covering,\n"
        s += "        zorder=mid,\n"
        s += "        linewidth=thin,\n"
        s += "    ),\n"
        s += "    capprops=dict(  # * Caps at the end of whiskers\n"
        s += "        alpha=covering,\n"
        s += "        zorder=mid,\n"
        s += "        linewidth=thick,\n"
        s += "    ),\n"
        s += ")\n"
        s += "\n"
        s += "### Stripplot kws\n"
        s += "strip_KWS = dict(\n"
        s += "    dodge=True,  # * Separates the points in hue\n"
        s += "    jitter=0.2,  # * How far datapoints of one group scatter across the x-axis\n"
        s += "    zorder=front,\n"
        s += "    ### Marker Style\n"
        s += "    alpha=0.5,\n"
        s += "    size=2,\n"
        s += "    # color='none',\n"
        s += "    edgecolor='white',\n"
        s += "    linewidth=thin,  # * Edge width of the marker\n"
        s += "    # facecolors='none',\n"
        s += ")\n"
        s += "\n"
        s += "###... PLOT\n"
        s += "(\n"
        s += "    DA.subplots() # ! Replace DA with your instance name \n"
        s += "    .fillaxes(kind='box', **box_KWS)\n"
        s += "    .fillaxes(kind='strip', **strip_KWS)\n"
        s += "    .edit_legend()\n"
        s += ")\n"
        s += "\n"

        pyperclip.copy(s)
        print("#! Code copied to clipboard, press ctrl+v to paste")
        return s


## !__________________________________________________________________________

# %% Matplotlib Runtime Config (RC)

mpl.rc("figure", dpi=250)

# %% get data

df, dims = ut.load_dataset("tips")  # * Tips
# df, dims = ut.load_dataset("fmri") # * FMRI

MP = MultiPlot(data=df, dims=dims)



# %%
