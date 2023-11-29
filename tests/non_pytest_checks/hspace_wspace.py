"""A script that shows that hspace and wspace are autoadjusted
"""
# %%

from matplotlib import pyplot as plt
import matplotlib as mpl
import plotastic as plst
from plotastic import utils as ut

print(mpl.__version__)

### Lower dpi
plt.rcParams["figure.dpi"] = 70


# %%
# == Example Data ======================================================
DF, dims = plst.load_dataset("tips", verbose=False)
DA = plst.DataAnalysis(DF, dims=dims)


# %%
def get_heights(DA: plst.DataAnalysis):
    ### Get bboxes of axes

    heights_cols = []
    for key, axes in DA.axes_iter__col_axes:
        heights = []
        # print(key)  # todo
        for ax in axes:
            bbox = ax.get_tightbbox()
            height = ut.get_bbox_height(bbox, in_inches=True)
            heights.append(height)

            # print(bbox.extents)  # todo
            # print(height)
            # print()
        heights_cols.append(heights)
    return heights_cols


if __name__ == "__main__":
    heights_cols = get_heights(DA)
    print(heights_cols)
    # print()


def adjust_hspace(DA: plst.DataAnalysis):
    height = DA.figsize[1]

    ### Adjust height to fit all axes
    heights_cols = get_heights(DA)
    heights = [sum(heights_col) for heights_col in heights_cols]
    new_height = max(heights)
    DA.fig.set_figheight(new_height, forward=True)

    ### That size increase stretched the axes, too, undo that
    # height_fraction = height / new_height
    # hspace = new_height - height
    # plt.subplots_adjust(hspace=hspace) #?? Doesn't work at all

if __name__ == "__main__":
    adjust_hspace(DA)
    print(DA.figsize)
    # print()

#%%
def adjust_hspace_recursive(DA: plst.DataAnalysis):
    """Since plt.adjust_subplots(hspace=...) doesn't work, we need to
    figure out something else. This function is a recursive approach
    that increases the figure height until all axes fit."""
    
    height = DA.figsize[1]

    ### Adjust height to fit all axes
    heights_cols = get_heights(DA)
    heights = [sum(heights_col) for heights_col in heights_cols]
    new_height = max(heights)
    
    ### Save some recursion steps
    if new_height / height > 1.7:
        new_height = new_height * 1.7
        # print("heightboost")
    print(new_height, height)
    
    ### Recursive increase
    #' Repeat until the sum of axes heights is less than the figure
    #' height. 99% of new_height is more than enough
    while 0.99 * new_height > height:
        #' Save some recursion steps, also makes nice spacing
        new_height = new_height * 1.1
        DA.fig.set_figheight(new_height, forward=True)
        height = DA.figsize[1]
        adjust_hspace_recursive(DA) #' Recursive call
        

    
if __name__ == "__main__":
    pass
    # DA.subplots(figsize=(5,1)).fillaxes(kind="strip", dodge=True)
    # adjust_hspace_recursive(DA)
    # print(DA.figsize)
    # print()



def check_hspace():
    for i in range(5):
        # print(labels)
        width=5
        height = i + 1

        ### Plot
        (
            DA.subplots(
                figsize=(width, height),
                # constrained_layout=True, # !! not working with subplots_adjust
                # hspace=.7,
            )
            .fillaxes(kind="strip", dodge=True)
            .edit_legend()
        )
        ### Spaces
        # adjust_hspace(DA)
        adjust_hspace_recursive(DA)
        
        ### Try mpl native functions
        # DA.fig.subplots_adjust(hspace=.9)
        # plt.subplots_adjust(hspace=.5)# ?? NOT WORKING AT ALL
        # DA.fig.tight_layout(pad=5.0)
        # DA.fig.tight_layout(h_pad=2)
        # plt.tight_layout(h_pad=2)
        # plt.subplot_tool()
        

        print()
        new_height = round(DA.figsize[1], 2)
        # plt.suptitle(f"width={width}, height={height, new_height}", y=1.1)
        
        # plt.close()


if __name__ == "__main__":
    plst.set_style("paper")
    # plst.set_style("classic")
    # plst.print_styles()
    check_hspace()


# def _get_legend_width(labels: list[str]) -> float:
#     """Calculates the width of the legend in inches, taking fontsize
#     into account"""

#     ### Add legend title, which is hue
#     labels = [DA.dims.hue] + labels  # TODO: replace with self

#     ### Split by new lines and flatten
#     labels = [label.split("\n") for label in labels]
#     labels = [item for sublist in labels for item in sublist]
#     # print(labels)

#     ### Get length of longest level (or title)
#     max_label_length = max([len(label) for label in labels])

#     ### Convert label length to inches
#     #' 1 inch = 72 points, one character = ~10 points
#     fontsize = _get_fontsize_legend()
#     character_per_inch = 72 / fontsize
#     if "Narrow" in DA.font_mpl:  # TODO: replace with self
#         character_per_inch = character_per_inch * 0.8

#     legend_width = max_label_length / character_per_inch

#     ### Add more for the markers
#     #' When the legend title (hue) is the largest, no space needed
#     if len(DA.dims.hue) != max_label_length:
#         # legend_width += 0.5 # TODO reactivate
#         print("added marker width")

#     return legend_width
