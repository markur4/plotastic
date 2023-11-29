"""A helper script that generates plots of different sizes to test the
things that highly depend on overall plot size, like legend positioning
"""
# %%

from matplotlib import pyplot as plt
import matplotlib as mpl
import plotastic as plst
from plotastic import utils as ut

### Lower dpi
plt.rcParams["figure.dpi"] = 70


# %%
# == Example Data ======================================================
DF, dims = plst.load_dataset("tips", verbose=False)
DA = plst.DataAnalysis(DF, dims=dims)



# %%
# == Utils: Legend Width ================================================

if __name__ == "__main__":
    ### Test with legend
    DA.plot().edit_legend()
    plt.close()
    legend = DA.legend
    legend_box = legend.get_tightbbox()
    legend_width = ut.get_bbox_width(legend_box)

    ### Test with axes
    box = DA.axes[0][0].get_tightbbox()
    box.extents
    fig_width = ut.get_bbox_width(box)


# %%
# == Legend ============================================================
"""
We want the legend to be `loc="center right"`. But that setting is bad,
we need to use bbox_to_anchor, otherwise the legend will be outside the
figure. However, With increasing figure width, the legend to drift away
from the figure. That effect is more drastic when bbox_to_anchor has
larger numbers than (1.0, 0.5). So we need to adjust borderaxespad
depending on figure width.
But that was also bad. I opted to stretch the figure instead and then
adjust the subplot size to fit the legend.
"""



def check_legend():
    label_list = [
        ["yes", "no"],
        ["ad", "saaaaaaaaaaaaaaa"],
        ["ad", "saaaaa\naaaaaaaaaa"],
    ]

    for i in range(20):
        for labels in label_list:
            # print(labels)
            width = i + 1

            ### Plot
            (DA.subplots(figsize=(width, 3)).fillaxes(kind="strip", dodge=True))

            DA.edit_legend(
                labels=labels,
                # borderaxespad=None,
                # loc="center right",
                # bbox_to_anchor=None,
            )
            ### Legend Positioning
            # _adjust_fig_to_fit_legend(DA, labels=labels)
            
            # print()
            plt.suptitle(f"width={width}", y=1.1)
            # plt.close()


if __name__ == "__main__":
    print(mpl.rcParams["legend.fontsize"])
    # plst.set_style("paper")
    plst.set_style("classic")
    # plst.print_styles()
    check_legend()



