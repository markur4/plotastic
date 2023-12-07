#
# %% imports

import seaborn as sns
import pandas as pd

import pytest


# import markurutils as ut
# import plotastic as plst
from plotastic import DataAnalysis

import DA_configs as dac


# %% Test per config


@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_ALL)
def test_normality(DF: pd.DataFrame, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True)
    DA.check_normality()


@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_ALL)
def test_homoscedasticity(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True)
    DA.check_homoscedasticity()


@pytest.mark.parametrize("DF, dims", dac.zipped_noempty_PAIRED)
def test_sphericity(DF, dims):
    DA = DataAnalysis(data=DF, dims=dims, verbose=True, subject="subject")
    DA.check_sphericity()


# %% interactive testing to display Plots

if __name__ == "__main__":
    import ipytest

    ipytest.run()
