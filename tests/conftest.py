"""Utilities for testing plotastic. Contains lists of arguments"""

# %% imports

from typing import Callable

import os
import warnings
from glob import glob


import pandas as pd

import matplotlib.pyplot as plt

import plotastic as plst

# import plotastic.utils.utils as ut
# import plotastic.utils.cache as utc
from plotastic.utils.subcache import SubCache

# %% 
### Cache it to speed up
MEMORY_TESTCONFIGS = SubCache(
    location="./tests",
    verbose=True,
    subcache_dir="testconfigs",
    assert_parent="tests",
)
### Clear cache if needed
# MEMORY_TESTCONFIGS.clear()


# %% Datasets

### Source of files is seaborn, markurutils just adds cut column

# ==
# == Load example data =================================================================

### Cache function
load_dataset = MEMORY_TESTCONFIGS.subcache(plst.load_dataset)

DF_tips, dims_tips = load_dataset("tips", verbose=False)
DF_fmri, dims_fmri = load_dataset("fmri", verbose=False)
DF_qpcr, dims_qpcr = load_dataset("qpcr", verbose=False)


# %% Arguments

### Args that make empty groups
dims_withempty_tips = [
    dict(y="tip", x="day", hue="sex", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="sex", hue="day", col="time"),
    dict(y="tip", x="sex", hue="day", row="time"),
    dict(y="tip", x="sex", hue="day", row="size-cut"),
    dict(y="tip", x="sex", hue="day"),
    dict(y="tip", x="sex"),
    dict(y="tip", x="size-cut"),
]


### Args making sure they don't make empty groups
# !! Don't add more, other tests assume 4 entries
dims_noempty_tips = [
    dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
    dict(y="tip", x="size-cut", hue="smoker", col="sex"),
    dict(y="tip", x="size-cut", hue="smoker"),
    dict(y="tip", x="size-cut"),
]

dims_noempty_fmri = [
    dict(y="signal", x="timepoint", hue="event", col="region"),
    dict(y="signal", x="timepoint", hue="region", col="event"),
    dict(y="signal", x="timepoint", hue="region"),
    dict(y="signal", x="timepoint", hue="event"),
    dict(y="signal", x="timepoint"),
]

dims_noempty_qpcr = [
    dict(y="FC", x="gene", hue="fraction", col="class", row="method"),
    dict(y="FC", x="gene", hue="fraction", col="method", row="class"),
    dict(y="FC", x="gene", hue="fraction", col="class"),
    dict(y="FC", x="gene", hue="fraction"),
    dict(y="FC", x="gene"),
]

# %%  Combine for pytest.parametrize

zipped_noempty_tips = [(DF_tips, dim) for dim in dims_noempty_tips]
zipped_noempty_fmri = [(DF_fmri, dim) for dim in dims_noempty_fmri]
zipped_noempty_qpcr = [(DF_qpcr, dim) for dim in dims_noempty_qpcr]


### Paired Data (with subject)
zipped_noempty_PAIRED = zipped_noempty_fmri + zipped_noempty_qpcr

### All should make 14 test
zipped_noempty_ALL = (
    zipped_noempty_tips + zipped_noempty_fmri + zipped_noempty_qpcr
)
# len(zipped_noempty_ALL) # * -> 14 total tests


# %% Dataanalysis objects


def get_DA_statistics(dataset: str = "qpcr") -> plst.DataAnalysis:
    """Makes a DA object with every possible data stored in it

    :param dataset: "tips", "fmri", or "qpcr"
    :type dataset: str
    """

    ### ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ### Example Data that's Paired, so we can use tests for paired data
        assert dataset not in ["tips"], f"{dataset} is not paired"

        ### Load example data
        DF, dims = plst.load_dataset(dataset, verbose=False)

        ### Init DA
        DA = plst.DataAnalysis(DF, dims, subject="subject", verbose=False)

        ### Assumptions
        DA.check_normality()
        DA.check_homoscedasticity()
        DA.check_sphericity()

        ### Omnibus
        DA.omnibus_anova()
        DA.omnibus_rm_anova()
        DA.omnibus_kruskal()
        DA.omnibus_friedman()

        ### Posthoc
        DA.test_pairwise()

    return DA


def get_DA_plot(dataset: str = "qpcr") -> plst.DataAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ### Load example data
        DF, dims = plst.load_dataset(dataset, verbose=False)

        ### Init DA
        DA = plst.DataAnalysis(DF, dims, subject="subject", verbose=False)

        DA.plot_box_strip()
        plt.close()
        return DA


def get_DA_all(dataset: str) -> plst.DataAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        DA = get_DA_statistics(dataset)
        DA.plot_box_swarm()
        plt.close()
        return DA


# %%
if __name__ == "__main__":
    pass
    # %%
    #!%%timeit
    # get_DA_statistics()

    # %%
    #!%%timeit
    # get_DA_plot()

    # %%
    #!%%timeit
    # get_DA_all(dataset="qpcr")

# %%

### Cache results of these functions to speed up testing
get_DA_statistics = MEMORY_TESTCONFIGS.cache(get_DA_statistics)
get_DA_plot = MEMORY_TESTCONFIGS.cache(get_DA_plot)
get_DA_all = MEMORY_TESTCONFIGS.cache(get_DA_all)

### Make DataAnalysis objects for testing
DA_STATISTICS: plst.DataAnalysis = get_DA_statistics("qpcr")
DA_PLOT: plst.DataAnalysis = get_DA_plot("qpcr")
DA_ALL: plst.DataAnalysis = get_DA_all("qpcr")


# %% Utils
###  (DF, dims) -> (DF, dims, kwargs)
def add_zip_column(zipped: list[tuple], column: list) -> list[tuple]:
    """Adds a column to a list of tuples. Useful for adding a list of arguments to a
    list of dataframes and dimensions. E.g.: (DF, dims) -> (DF, dims, kwargs)

    :param zipped: A set of dataframes and dimensions in this shape [(df, dim), (df,
        dim), ...] ready to be used in pytest.parametrize
    :type zipped: list[tuple]
    :param column: A list of ne arguments to be added to each tuple in zipped. Must be same length as zipped
    :type column: list
    :return: (DF, dims) -> (DF, dims, kwargs)
    :rtype: list[tuple]
    """

    assert len(zipped) == len(column), "zipped and column must be same length"

    zipped_with_column = []
    for tup, e in zip(zipped, column):
        zipped_with_column.append(tup + (e,))
    return zipped_with_column


def cleanfiles(fname: str):
    """deletes all files that start with fname"""
    testfiles = glob(fname + "*")
    for file in testfiles:
        os.remove(file)
