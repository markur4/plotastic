#
# %%

import markurutils as ut
import plotastic as plst
from plotastic.multiplot import MultiPlot

import matplotlib as mpl


# %% Matplotlib Runtime Config (RC)

mpl.rc("figure", dpi=250)

# %% get data

df, dims = ut.load_dataset("tips")  # * Tips
# df, dims = ut.load_dataset("fmri") # * FMRI

MP = MultiPlot(data=df, dims=dims)


# %% Test boxplot_strip

MP = MP.plot_box_strip(marker_size=4, strip_kws=dict(alpha=0.3, linewidth=0.3))

# %% Test boxplot_strip_SNIP
s = MP.plot_box_strip_SNIP()

# # Docs Boxplot:        https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot
# # Docs Stripplot:      https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot

# ### ... PARAMETERS
# ### Linewidths
# thin, thick = 0.3, 1.0
# ### Alpha
# covering, translucent, hazy = 1.0, .5, .3
# ### z-order
# front, mid, background, hidden = 100, 50, 1, -1

# ### ... KEYWORD ARGUMENTS
# ### Boxplot kws
# box_KWS = dict(
#     showfliers=False,
#     boxprops=dict(  # * Box line and surface
#         alpha=hazy,
#         linewidth=thin,
#     ),
#     medianprops=dict(  # * Median line
#         alpha=covering,
#         zorder=front,
#         linewidth=thick,
#     ),
#     whiskerprops=dict(  # * Lines conencting box and caps
#         alpha=covering,
#         zorder=mid,
#         linewidth=thin,
#     ),
#     capprops=dict(  # * Caps at the end of whiskers
#         alpha=covering,
#         zorder=mid,
#         linewidth=thick,
#     ),
# )

# ### Stripplot kws
# strip_KWS = dict(
#     dodge=True,  # * Separates the points in hue
#     jitter=0.2,  # * How far datapoints of one group scatter across the x-axis
#     zorder=front,
#     ### Marker Style
#     alpha=0.5,
#     size=2,
#     # color='none',
#     edgecolor='white',
#     linewidth=thin,  # * Edge width of the marker
#     # facecolors='none',
# )

# ###... PLOT
# (
#     DA.subplots() # ! Replace DA with your instance name
#     .fillaxes(kind='box', **box_KWS)
#     .fillaxes(kind='strip', **strip_KWS)
#     .edit_legend()
# )
