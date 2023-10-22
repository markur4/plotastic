# %% Imports
import pytest

import plotastic as plst

import conftest as ct


# %% test dendrogram
@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_levels_dendrogram(DF, dims):
    ### No need to evaluate level combos if there's just an X and no facet (hue, col, row)
    if not len(dims.keys()) == 2:
        DA = plst.DataAnalysis(data=DF, dims=dims)
        DA.levels_dendrogram()


# %% test combocounts


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_levels_combocounts(DF, dims):
    ### No need to evaluate level combos if there's just an X and no facet (hue, col, row)
    if not len(dims.keys()) == 2:
        DA = plst.DataAnalysis(data=DF, dims=dims)
        DA.levels_combocounts()


if __name__ == "__main__":
    import pandas as pd

    DF, dims = plst.load_dataset("qpcr", verbose=False)
    DF, dims = plst.load_dataset("tips", verbose=False)
    DF, dims = plst.load_dataset("fmri", verbose=False)

    ### Init DataAnalysis object
    DA = plst.DataAnalysis(data=DF, dims=dims)

    DA._count_levelcombos()

    DA.levelkeys
    len(DA.levelkeys)
    DA.levels_combocounts()
    DA.levels_dendrogram()

# %% run interactively

if __name__ == "__main__":
    import ipytest

    ipytest.run()


# %%
