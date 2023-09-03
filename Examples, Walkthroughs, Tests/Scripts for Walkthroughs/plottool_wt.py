#
# %% Imports
import matplotlib.pyplot as plt


import plotastic as plst
import markurutils as ut

from plotastic.plotting.plottool import PlotTool

# %% Quick testing . . . . . . . . . . . . . . . . . . . . . . .

DF, dims = ut.load_dataset("tips")
DF2, dims2 = ut.load_dataset("fmri")
DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
PT = PlotTool(data=DF, dims=DIMS).switch("x", "col")
# PT2 = PlotTool(data=DF2, dims=DIMS2).switch("x", "col")

PT.data_describe()
PT.plot()


# %% Detailed Testing

DF, dims = ut.load_dataset("tips")  # * Load Data. Dims
DIMS = dict(y="tip", x="day", hue="sex", col="smoker", row="time")
PT = PlotTool(
    data=DF, dims=DIMS
)  # ! We use PT, but please access these functions via the DataAnalysis class (DA)

### Test Parts
PT.subplots()  # * Make Figure and Axes

### Test Interface functions
PT = (
    PT.plot()
)  # ! Make sure to return new instance of PlotTool, otherwise subsequent edits won't apply

### Plot in two steps
PT.subplots()  # * Make empty axes
PT.fillaxes(kind="swarm")  # * Fill axes with seaborn graphics
plt.show()

### Plot in One step
PT.plot(kind="bar")

### Access axes via PT.axes (of type np.ndarray[mpl.axes._subplots.AxesSubplot])
for ax in PT.axes.flatten():
    ax.set_title("bla")

### Overlay two plots:
PT.subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
PT.fillaxes(kind="box", boxprops=dict(alpha=0.5))
PT.fillaxes(kind="swarm", size=3, dodge=True)

for i, ax in enumerate(PT.axes.flatten()):  # * Pick axes as you want!
    if i == 2:
        ax.set_title("THIRD!")

### Use Chaining
# * (methods act inplace, so you have to re-initialize PT to start from scratch!)
PT: PlotTool = (  # * Needs to be passed in order to make it modifyable in subequent lines
    PT.switch("x", "col")  # * Experiment with switching dimensions!
    .subplots(sharey=True, gridspec_kw=dict(wspace=0.2, hspace=0.5))
    .fillaxes(kind="box", boxprops=dict(alpha=0.5))
    .fillaxes(kind="swarm", size=3, dodge=True)
    .edit_legend()  # * Add standard legend
)

for i, ax in enumerate(PT.axes.flatten()):
    if i == 2:
        ax.set_title("THIRD!")

### Don't memorize this, just copy code to the clipboard!
PT.subplots_SNIP(doclink=True)
DA = PT  # ! If you use DataAnalysis of PlotTool, it makes no difference!
# !(We use DA, since its' not intended to use PT directly)

### There's a snippet for fillaxes too!
PT.fillaxes_SNIP(kind="bar", doclink=True)
# . . . https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot #
# kws = dict(alpha=.8)
# for ax, df in DA.iter_axes_and_data:
#     sns.barplot(data=df, ax=ax, y='tip', x='day', hue='sex', **kws)
#     ax.legend_.remove()
# DA.edit_legend()  # * Add legend to figure

### A snippet for configuring Legends
PT.edit_legend_SNIP()
# # . . . https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.legend #
# DA.fig.legend(
#     title='sex', #* Hue factor
#     handles=DA.legend_handles_and_labels[0],
#     labels=DA.legend_handles_and_labels[1],
#     loc='center right', #* Rough location
#     bbox_to_anchor=(1.15, 0.50), #* Exact location in width, height relative to complete figure
#     ncols=1, #* If >1, labels are displayed next to each other
#     borderaxespad=3, #* Padding around axes, (pushing legend away)
#     markerscale=1.5, #* Marker size relative to plotted datapoint
#     frameon=False, #* Remove frame around legend
# )

# ! Snippets use matplotlib functions, which don't return PlotHelper object, so they can NOT be chained!
# ! Use them at the end of a layer chain!

### Try Different Dataset
DF, dims = ut.load_dataset("fmri")
PT = PlotTool(data=DF, dims=dims)

PT.plot()
PT.switch("row", "col").plot()
PT.set(row="none", col="none").plot()

### Logarithmic scale
PT.edit_y_scale_log(base=2)

### Snippet for Logarithmic scaling
PT.edit_xy_scale_SNIP(doclink=True)
# # . . . https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale #
# for ax in DA.axes.flatten():
#     ax.set_yscale('log',  # * 'symlog', 'linear', 'logit',
#         base=10,
#         nonpositive='clip', # * 'mask': masked as invalid, 'clip': clipped to a very small positive number
#     )
#     # ax.set_xscale('log') # ? Rescale x-axis

### Gridlines
PT.edit_grid()

### Snippet for Gridlines
PT.edit_grid_SNIP()
# for ax in PT.axes.flatten():
#     ax.yaxis.grid(True, which='major', ls='-', linewidth=0.5, c='grey')
#     ax.yaxis.grid(True, which='minor', ls='-', linewidth=0.2, c='grey')
#     ax.xaxis.grid(True, which='major', ls='-', linewidth=0.3, c='grey')

### Show minor tick-labels
PT.edit_y_ticklabels_log_minor(subs=[2, 3, 5, 7])

### Snippet for minor tick-labels
PT.edit_y_ticklabels_log_minor()
# for ax in PT.axes.flatten():
#     #* Set minor ticks, we need ScalarFormatter, others can't get casted into float
#     ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter(useOffset=0, useMathText=False))
#     #* Iterate through labels
#     for label in ax.yaxis.get_ticklabels(which='minor'):
#         # ? How else to cast float from mpl.text.Text ???
#         label_f = float(str(label).split(', ')[1])  #* Cast to float
#         mantissa = int(round(ut.mantissa_from_float(label_f))) #* Calculate mantissa
#         if not mantissa in [2, 3, 5, 7]:
#             label.set_visible(False) # * Set those not in subs to invisible

### Change Font Sizes
PT.edit_fontsizes(ticklabels=9, xylabels=10, axis_titles=11)

### Snippet for Font Sizes
PT.edit_fontsizes_SNIP()
# ticklabels, xylabels, axis_titles = 9, 10, 11 ### <--- CHANGE THIS [pt]
# for ax in PT.axes.flatten():
#     ax.tick_params(axis='y', which='major', labelsize=ticklabels) # * Ticklabels
#     ax.tick_params(axis='y', which='minor', labelsize=ticklabels-.5)
#     ax.tick_params(axis='x', which='major', labelsize=ticklabels)
#     ax.tick_params(axis='x', which='minor', labelsize=ticklabels-.5)
#     ax.yaxis.get_label().set_fontsize(xylabels) # * xy-axis labels
#     ax.xaxis.get_label().set_fontsize(xylabels)
#     ax.title.set_fontsize(axis_titles) # * Title

### Format your axes titles:
PT.edit_titles_with_func(
    row_func=lambda x: x.upper(),
    col_func=lambda x: "",
    connect=" || ",
)

### Snippet for axes title formatting
PT.edit_titles_with_func_SNIP()
# row_format = lambda x: x.upper() #* e.g. try lambda x: x.upper()
# col_format = lambda x: x
# connect = '\n' #* newline. Try ' | ' as a separator in the same line
# for rowkey, axes in PT.axes_iter__row_axes:
#     for ax in axes:
#         title = row_format(rowkey)
#         ax.set_title(title)
# for colkey, axes in PT.axes_iter__col_axes:
#     for ax in axes:
#         title = ax.get_title() + connect + col_format(colkey)
#         ax.set_title(title)

### Change the axis titles fully manually
PT.edit_title_replace(
    [
        "guccy",
        "guccy-gu",
        "dada\nnana",
        "is this empty?",
        "next row",
        "...",
    ]
)

### Snippet for replacing titles
PT.edit_title_replace_SNIP()
# titles = ['Lunch \nThsdfr ', 'Lunch \nFri ', 'Lunch \nSat ', 'Lunch \nSun ', 'Dinner \nThur ', 'Dinner \nFri ', 'Dinner \nSat ', 'Dinner \nSun ']
# for ax, title in zip(PT.axes.flatten(), titles):
#     ax.set_title(title)

### Change the axis labels
PT.edit_xy_axis_labels(
    leftmost_col="tipdf",
    notleftmost_col="",
    lowest_row="smoker",
    notlowest_row="d",
)

### Snipper for changing the axis labels
PT.edit_xy_axis_labels_SNIP()
# ### y-axis labels
# for ax in PT.axes_iter_leftmost:
#     ax.set_ylabel('tip')
# for ax in PT.axes_iter_notleftmost:
#     ax.set_ylabel('')
# ### x-axis labels
# for ax in PT.axes_iter_lowerrow:
#     ax.set_xlabel('smoker')
# for ax in PT.axes_iter_notlowerrow:
#     ax.set_xlabel('')

### Edit xtick-labels
PT.edit_x_ticklabels(
    lowerrow=["NOO", "yup"],
    notlowerrow=["dd", "ee"],
    rotation=0,
    ha="center",
    pad=1,
)
### Snippet for changing the axis labels
PT.edit_x_ticklabels_SNIP()
# DA=PT
# notlowerrow = ['', '']
# lowerrow = ['Yes', 'No']
# kws = dict(
#     rotation=30, #* Rotation in degrees
#     ha='right', #* Horizontal alignment [ 'center' | 'right' | 'left' ]
#     va='top', #* Vertical Alignment   [ 'center' | 'top' | 'bottom' | 'baseline' ]
# )
# ticks = [0, 1]
# for ax in DA.axes_iter_notlowerrow:
#     ax.set_xticks(ticks=ticks, labels=notlowerrow, **kws)
#     ax.tick_params(axis='x', pad=.01) #* Sets distance to figure
# for ax in DA.axes_iter_lowerrow:
#     ax.set_xticks(ticks=ticks, labels=lowerrow, **kws)
#     ax.tick_params(axis='x', pad=.01) #* Sets distance to figure


# %% Automatic Testing ................................................................................


def tester(DF, dims):
    PT = PlotTool(data=DF, dims=dims, verbose=True)  # .switch("x", "col")

    PT: PlotTool = (
        PT.subplots(sharey=True)
        .fillaxes(kind="box", boxprops=dict(alpha=0.5))
        .fillaxes(kind="swarm", size=3, dodge=True)
        .edit_axtitles_reset()
    )
    PT = (
        PT.edit_titles()
        .edit_titles_with_func()
        .edit_xy_axis_labels()
        .edit_y_scale_log(base=10)
        .edit_y_ticklabel_percentage()
        .edit_y_ticklabels_log_minor(subs=[2, 3, 5, 7])
        .edit_x_ticklabels()
        .edit_grid()
        .edit_fontsizes(9, 10, 7)
        # .edit_replace_titles(titles = ["1", "2", "3", "4", "5", "6", "7", "8"])
    )
    if PT.dims.hue:  # * Only when legend
        PT = PT.edit_legend()
    # plt.close()


dimses = [
    dict(y="tip", x="day", hue="sex", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="sex", hue="day", col="time"),
    dict(y="tip", x="sex", hue="day", row="time"),
    dict(y="tip", x="sex", hue="day", row="size-cut"),
    dict(y="tip", x="sex", hue="day"),
    dict(y="tip", x="sex"),
    dict(y="tip", x="size-cut"),
]

DF, dims = ut.load_dataset("tips")
for dim in dimses:
    print("\n !!!", dim)
    tester(DF, dim)


# %%
