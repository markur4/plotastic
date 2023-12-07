#
# %% imports

import warnings

import seaborn as sns
import pandas as pd

import pytest


# import markurutils as ut
# import plotastic as plst
from plotastic import DataAnalysis

import DA_configs as dac


# %% Test per config


@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_ALL)
def test_omnibus_anova(DF: pd.DataFrame, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True)
    DA.omnibus_anova()


# !! Warnings happen when groups have only one sample
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_PAIRED)
def test_omnibus_rm_amova(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, subject="subject", verbose=True)
    DA.omnibus_rm_anova()


@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_ALL)
def test_omnibus_kruskal(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True)
    DA.omnibus_kruskal()


# @pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_PAIRED)
def test_omnibus_friedman(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, subject="subject", verbose=True)
    DA.omnibus_friedman()


# %% interactive testing to display Plots

if __name__ == "__main__":
    import ipytest

    ipytest.run()
