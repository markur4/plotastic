#
# %% 
### Imports

import matplotlib as mpl


# %%
# == Variables ro reuse ================================================
FONTSIZE = 10


# %%
# == STYLE PAPER =======================================================
PAPER = {
    ### Figure
    "figure.dpi": 200,  #' Displaying figures doesn't need as much dpi as saving them
    "figure.figsize": (3, 3),  #' default is way too big
    # "figure.facecolor": "gray",  #' it's easier on the eyes
    ### Savefig
    "savefig.dpi": 300,  #' Saving figures needs more dpi
    "savefig.format": "pdf",
    # "savefig.transparent": True,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    ### Font
    "font.family": "sans-serif",
    "font.sans-serif": "Arial Narrow",
    "font.size": FONTSIZE,
    "font.weight": "bold",
    # ## Lines
    "lines.linewidth": 0.75,
    # ## Axes
    "axes.spines.right": True,  #' requires argument despine=False
    "axes.spines.top": True,
    "axes.linewidth": 0.75,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlepad": 5,
    "axes.labelsize": FONTSIZE,  #' fontsize of the x any y labels
    # ## Grid
    # "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    # ## Ticks
    "ytick.left": True,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "ytick.major.pad": 0.9,  #' distance Yticklabels and yticks
    "ytick.minor.pad": 0.8,
    "xtick.major.pad": 2,  #' distance Xticklabels and yticks
    "xtick.minor.pad": 2,
    "ytick.major.size": 2.5,
    "ytick.minor.size": 2,
    "xtick.major.size": 2.5,
    "xtick.minor.size": 2,
    # ## Legend
    "legend.fancybox": False,  #' use rounded box for legend
    "legend.title_fontsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "legend.markerscale": 1.3,  #' size scaled of markers in legend
    "legend.handleheight": 0.7,  #' line distance between legend entries
    "legend.handletextpad": 0.1,  #' distance markers legend text
    # 'legend.borderaxespad': 1, #' distance legend axes border, must be negative..?
    "legend.borderpad": 0.001,
    # 'text.usetex': True,
    # 'scatter.marker': 'x',
}

# == Collect STYLES ====================================================

### Give styles a name and add them to STYLES_PLST
STYLES = {
    "default": PAPER,
    "paper": PAPER,
}

### Keys are the styles, values are the keys of the styles
STYLENAMES = {
    "plotastic": sorted(list(STYLES.keys())),
    "seaborn": ["white", "dark", "whitegrid", "darkgrid", "ticks"],
    "matplotlib": mpl.style.available,
}



