#
# %%
import matplotlib as mpl

from plotastic import utils as ut

import DA_configs as dac


# %%
def test_font_functions():
    ut.mpl_font()
    ut.mpl_fontsizes_get_all()
    ut.mpl_fontsize_from_rc(rc_param="legend.fontsize")

    ut.mpl_fontpath()


if __name__ == "__main__":
    test_font_functions()

    # mpl.rcParams["font.size"] = 22
    # print(mpl.rcParams["font.size"]) #' returns an integer
    # print(mpl.rcParams["legend.fontsize"]) #' returns mediu
    # m

    ### Fontsizes
    d = ut.mpl_fontsizes_get_all()
    print(d)

    fs = ut.mpl_fontsize_from_rc()
    legend_fs = ut.mpl_fontsize_from_rc(rc_param="legend.fontsize")
    print(legend_fs)

    ### Font
    # plst.set_style("paper")
    font = ut.mpl_font
    fontpath = ut.mpl_fontpath()
    print(fontpath)

    if "Narrow" in font:
        print("narrow")


# %%
def test_get_bbox_width():
    DA = dac.DA_ALL
    # DA.legend.get_window_extent()
    bbox = DA.legend.get_tightbbox()
    ut.get_bbox_width(bbox)
