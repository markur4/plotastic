#
# %% Imports

import matplotlib as mpl

import markurutils as ut
import plotastic as plst


# %% Some info on matplotlib rc files

### How Plotastic changes your rc settings
"""Basically,  plst iterates through dictionaries stored in plst.rc executing mpl.rcParams['setting'] = value, e.g.:

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

# %% Check where matplotlib is looking for rc files
print(mpl.get_configdir())  # * This prints a directory where no rc file is found

print(mpl.matplotlib_fname())  # * This seems more correct


# %% Default Style

df, dims = ut.load_dataset("fmri")
DA = plst.DataAnalysis(df, dims)


### Default plot
plst.rc.set_style("default")
DA.catplot()

# %% Apply Style

plst.rc.set_style("paper")
g = DA.catplot()


# %% Apply Palette

plst.rc.set_palette(verbose=True)  # * defaults to "Paired"
g = DA.catplot()

# %% Apply different palette
plst.rc.set_palette(
    "bright", verbose=False
)  # * pick another color, suppress demonstration of colors
g = DA.catplot()

g.savefig("test.pdf")