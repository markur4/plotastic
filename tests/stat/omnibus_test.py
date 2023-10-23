#
# %% imports

import warnings

import seaborn as sns
import pandas as pd

import pytest


# import markurutils as ut
# import plotastic as plst
from plotastic import DataAnalysis

import conftest as ct




# %% Test per config


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_omnibus_anova(DF: pd.DataFrame, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True)
    DA.omnibus_anova()

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_PAIRED)
def test_omnibus_rm_amova(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, subject="subject", verbose=True)
    DA.omnibus_rm_anova()


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_omnibus_kruskal(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True)
    DA.omnibus_kruskal()


# %% interactive testing to display Plots

if __name__ == "__main__":
    import ipytest
    ipytest.run()
