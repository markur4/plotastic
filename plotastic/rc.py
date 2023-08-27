#
# %% Imports

import matplotlib as mpl
from matplotlib.pyplot import grid
from numpy import size


# %% .... Apply settings ........................................................
def set_style(settings: dict | str):
    if isinstance(settings, str):
        if settings == "paper":
            settings = STYLE_PAPER
        elif settings in ("default", "defaults"):
            mpl.rcParams.update(mpl.rcParamsDefault)
            return None
        else:
            raise ValueError(
                f"Unknown settings string {settings}. " f"Use one of: {['plst_paper']}"
            )

    ### Define settings
    for setting, value in settings.items():
        mpl.rcParams[setting] = value


# %% .... Style: Paper .............................................................

FONTSIZE = 10

STYLE_PAPER = {
    # ... Figure
    "figure.dpi": 300,
    "figure.figsize": (1, 1),  # * default is way too big
    # ... Font
    "font.family": "sans-serif",
    "font.sans-serif": "Arial Narrow",
    "font.size": FONTSIZE,
    "font.weight": "bold",
    # ... Lines
    "lines.linewidth": 0.75,
    # ... Axes
    "axes.spines.right": True,  # * requires argument despine=False
    "axes.spines.top": True,
    "axes.linewidth": 0.75,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlepad": 5,
    "axes.labelsize": FONTSIZE,  # * fontsize of the x any y labels
    # ... Grid
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    # ... Ticks
    "ytick.left": True,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "ytick.major.pad": 0.9,  # * distance Yticklabels and yticks
    "ytick.minor.pad": 0.8,
    "xtick.major.pad": 2,  # * distance Xticklabels and yticks
    "xtick.minor.pad": 2,
    "ytick.major.size": 2.5,
    "ytick.minor.size": 2,
    "xtick.major.size": 2.5,
    "xtick.minor.size": 2,
    # ... Legend
    "legend.fancybox": False,  # * use rounded box for legend
    "legend.title_fontsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "legend.markerscale": 1.3,  # * size scaled of markers in legend
    "legend.handleheight": 0.7,  # * line distance between legend entries
    "legend.handletextpad": 0.1,  # * distance markers legend text
    # 'legend.borderaxespad': 1, # * distance legend axes border
    "legend.borderpad": 0.001,
    # 'text.usetex': True,
    # 'scatter.marker': 'x',
}
