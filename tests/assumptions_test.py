#
# %% imports

import seaborn as sns
import pandas as pd

import pytest
import ipytest


# import markurutils as ut
# import plotastic as plst
from plotastic import Assumptions

import conftest as ct


# %% Test per config


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_all)
def test_normality(DF: pd.DataFrame, dims):
    AS = Assumptions(data=DF, dims=dims, verbose=False)
    AS.check_normality()


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_all)
def test_homoscedasticity(DF, dims):
    AS = Assumptions(data=DF, dims=dims, verbose=False)
    AS.check_homoscedasticity()

#%% interactive testing to display Plots

if __name__ == "__main__":
    ipytest.run()