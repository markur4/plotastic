#
# %% Imports

import matplotlib as mpl
from matplotlib.pyplot import grid
from numpy import size


# %% .... Apply settings ........................................................
def mplstyle_set(settings: dict | str):
    if isinstance(settings, str):
        if settings == "paper":
            settings = SETTINGS_PLST_PAPER
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


# %% .... Settings .............................................................

FONTSIZE = 10

SETTINGS_PLST_PAPER = {
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


# %% Some info on matplotlib rc files

### How Plotastic changes your rc settings
"""Basically, plst iterates through dictionaries stored in plst.rc executing mpl.rcParams['setting'] = value, e.g.:

    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["lines.linewidth"] = 0.75
    mpl.rcParams["axes.linewidth"] = 0.75
    mpl.rcParams["axes.labelweight"] = "bold"
    ... etc.

Hence, no styles are permanently changed. You can always reset your rc settings by restarting your kernel or executing mpl.rcParams.update(mpl.rcParamsDefault).
The reason for not working with .mplstyle files is that I don't know how to influence the fontsize-increments from relative sizes (large, large-x, etc.).
"""

### About matplotlib rc files
"""From: https://matplotlib.org/stable/tutorials/introductory/customizing.html

Matplotlib looks for matplotlibrc in four locations, in the following order:
1. matplotlibrc in the current working directory, usually used for specific customizations that you do not want to apply elsewhere.
2. $MATPLOTLIBRC if it is a file, else $MATPLOTLIBRC/matplotlibrc.
3. It next looks in a user-specific place, depending on your platform:
    On Linux and FreeBSD, it looks in .config/matplotlib/matplotlibrc (or $XDG_CONFIG_HOME/matplotlib/matplotlibrc) if you’ve customized your environment.
    On other platforms, it looks in .matplotlib/matplotlibrc.
    https://matplotlib.org/stable/users/faq/troubleshooting_faq.html#locating-matplotlib-config-dir
4. It looks in the installation directory (usually something like /usr/local/lib/python3.9/site-packages/matplotlib/mpl-data/matplotlibrc, but note that this depends on the installation).
"""


# %% test

import markurutils as ut
from plotastic.dataanalysis import DataAnalysis
import seaborn as sns


df, dims = ut.load_dataset("fmri")
DA = DataAnalysis(df, dims)


### Default plot
mplstyle_set("default")
DA.catplot()

# %% Apply settings


mplstyle_set(SETTINGS_PLST_PAPER)
g = DA.catplot()


# %% Check where matplotlib is looking for rc files
print(mpl.get_configdir())  # * This prints a directory where no rc file is found

print(mpl.matplotlib_fname())  # * This seems more correct
