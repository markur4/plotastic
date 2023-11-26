# %%

import pytest
import ipytest
import matplotlib.pyplot as plt

import plotastic as plst

import _DA_configs as dac


# %%
titles_zip = [
    {("Lunch", "Male"): "blaa"},
    {("Male"): "blAA"},
    None,
    None,
]
zipped_tips = dac.add_zip_column(dac.zipped_noempty_tips, titles_zip)


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

titles_zip = [
    ["sdfsfd", None, "dd", None],
    [None, "aa"],
    None,
    None,
]
zipped_tips = dac.add_zip_column(dac.zipped_noempty_tips, titles_zip)


@pytest.mark.parametrize("DF, dims, titles", zipped_tips)
def test_edit_titles_replace(DF, dims, titles: dict):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    if DA.factors_rowcol:  #' Need facetting, otherwise no axes
        DA.plot()
        DA.edit_titles_replace(titles=titles)
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()

#%%

