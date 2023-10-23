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


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_normality(DF: pd.DataFrame, dims):
    AS = Assumptions(data=DF, dims=dims, verbose=True)
    AS.check_normality()


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_homoscedasticity(DF, dims):
    AS = Assumptions(data=DF, dims=dims, verbose=True)
    AS.check_homoscedasticity()


@pytest.mark.parametrize("DF, dims", ct.zipped_noempty_ALL)
def test_sphericity(DF, dims):
    ### Don't test sphericity for unpaired data
    if not "total_bill" in DF.columns:
        AS = Assumptions(data=DF, dims=dims, verbose=True, subject="subject")
        AS.check_sphericity()


# %% interactive testing to display Plots

if __name__ == "__main__":
    ipytest.run()
