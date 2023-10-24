#
# %% imports
import os

import pytest

# import seaborn as sns
# import pandas as pd

# import markurutils as ut
import plotastic as plst

import conftest as ct


# %% Test


def test_save_statistics():
    """Test export_statistics()"""
    # %% Import Data
    DA = ct.get_DA_with_full_statistics()

    # %% test
    out = "test_result"
    DA.save_statistics(fname=out)

    ### Delete file
    os.remove(out + ".xlsx")


# %% interactive testing to display Plots

if __name__ == "__main__":
    import ipytest

    ipytest.run()
