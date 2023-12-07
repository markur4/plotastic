#
# %%

import pytest
import ipytest
import matplotlib.pyplot as plt

import plotastic as plst

import DA_configs as dac


# %%
titles_tips = [
    {("Lunch", "Male"): "blaa"},
    {("Male"): "blAA"},
    None,
    None,
]
zipped_tips = dac.add_zip_column(dac.zipped_noempty_tips, titles_tips)


@pytest.mark.parametrize("DF, dims, axtitles", zipped_tips)
def test_edit_titles(DF, dims, axtitles: dict):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    if DA.factors_rowcol:  #' Need facetting, otherwise no axes
        DA.plot()
        DA.edit_titles(axtitles=axtitles)
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_titles_with_func(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    if DA.factors_rowcol:  #' Need facetting, otherwise no axes
        DA.plot().edit_titles_with_func(
            row_func=lambda x: x.upper(),
            col_func=lambda x: "hä",
            connect=" || ",
        )

    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()

# %%
titles_tips = [
    ["sdfsfd", None, "dd", None],
    [None, "aa"],
    None,
    None,
]
zipped_tips = dac.add_zip_column(dac.zipped_noempty_tips, titles_tips)


@pytest.mark.parametrize("DF, dims, titles", zipped_tips)
def test_edit_titles_replace(DF, dims, titles: dict):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    if DA.factors_rowcol:  #' Need facetting, otherwise no axes
        (DA.plot().edit_titles_replace(titles=titles))
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_xy_axis_labels(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot().edit_xy_axis_labels(
        x="ui!",
        x_lowest_row="rambazamba",
        x_notlowest_row="FLOH",
        y="Johannes",
        y_leftmost_col="Gertrude",
        y_notleftmost_col="Hä?",
    )
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_y_scale_log(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot().edit_y_scale_log(base=2)
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_y_ticklabel_percentage(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot().edit_y_ticklabel_percentage(
        decimals_major=1,
        decimals_minor=1,  # !! Not working
    )
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_y_ticklabels_log_minor(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot().edit_y_scale_log(base=2).edit_y_ticklabels_log_minor(
        subs=[2, 3, 5, 7],
    )
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
labels_zip = [
    ["sdfsfd", "dddd"],
    ["sdfsfd", "dddd"],
    ["sdfsfd", "dddd"],
    ["sdfsfd", "dddd"],
]
zipped_tips = dac.add_zip_column(dac.zipped_noempty_tips, labels_zip)


@pytest.mark.parametrize("DF, dims, labels", zipped_tips)
def test_edit_x_ticklabels_exchange(DF, dims, labels):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot().edit_x_ticklabels_exchange(
        labels=labels,
        labels_lowest_row=[l.upper() for l in labels],
    )
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_x_ticklabels_exchange(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot().edit_x_ticklabels_rotate(
        rotation=75,
        ha="center",
        # va="top",
        pad=0.1,
    )
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()

# %%
plt.close("all")


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_grid(DF, dims):
    plt.close()
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    (
        DA.plot()
        .edit_y_scale_log(base=2)  #' To see minor ticks
        .edit_grid(
            y_major_kws=dict(ls="--", linewidth=0.5, c="grey"),
            y_minor_kws=dict(ls=":", linewidth=0.2, c="grey"),
            x_major_kws=dict(ls="--", linewidth=0.6, c="grey"),
        )
    )
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_legend(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    if DA.dims.hue:
        DA.plot().edit_legend(
            reset_legend=True,
            title="HUI",
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
            borderaxespad=1,
            # pad=0.5,
            frameon=True,
        )  #' To see minor ticks

    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_fontsizes(DF, dims):
    plt.close()
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)

    DA.plot().edit_fontsizes(
        ticklabels=14,
        xylabels=16,
        axis_titles=18,
    )  #' To see minor ticks

    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()
