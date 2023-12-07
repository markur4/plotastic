#
# %% imports

import pytest

import pandas as pd

import os
from glob import glob
from pathlib import Path

import plotastic as plst

import DA_configs as dac


# %%
import IPython

IPython.extract_module_locals()[1].get("__vsc_ipynb_file__")


# %% Test

DF, dims = plst.load_dataset("tips", verbose=False)
DA = plst.DataAnalysis(DF, dims, verbose=False)
DA_COMPLETE = dac.DA_STATISTICS


# %% Test prevent_overwrite


def test_prevent_overwrite():
    ### Define a name
    testfile_name = "_FILE_123"
    distraction_names = [
        "_FILE_",
        "__FILE_",
        "_FILE_12",
        "_FIL_12",
    ]

    def mk_testfiles(testfile_name) -> str:
        ### Make a testfile excel
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_excel(testfile_name + ".xlsx")

        ### Make a testfile text as a distraction
        with open(testfile_name + ".txt", "w") as f:
            f.write("I am an text file")

        return testfile_name

    ### Cleanup before testing
    dac.cleanfiles(testfile_name)
    for name in distraction_names:
        dac.cleanfiles(name)

    ### Make Distraction Files
    for name in distraction_names:
        mk_testfiles(name)

    # == TEST 0: mode="day"
    kws = dict(overwrite="day")
    new = DA.filer.prevent_overwrite(testfile_name, **kws)
    assert (
        new == testfile_name + f"_{DA.filer.current_day}"
    ), f"new_name = {new}, testfile_name = {testfile_name}"

    # == TEST 1: mode="nothing"
    kws = dict(overwrite="nothing")

    ### If NO file exists, it should return the same name with _0
    new = DA.filer.prevent_overwrite(testfile_name, **kws)
    assert (
        new == testfile_name + "_0"
    ), f"new_name = {new}, testfile_name = {testfile_name}"

    ### If a file EXISTS, it should return the same name with _0
    tested = mk_testfiles(testfile_name)
    new = DA.filer.prevent_overwrite(testfile_name, **kws)
    assert (
        new == testfile_name + "_0"
    ), f"new_name = {new}, testfile_name = {tested}"

    ### If a file with _0 exists, it should return a new name with _1
    tested = mk_testfiles(new)  #' "testfile_name_0"
    new = DA.filer.prevent_overwrite(testfile_name, **kws)
    assert (
        new == testfile_name + "_1"
    ), f"new_name = {new}, testfile_name = {tested}"

    ### If a file with _1 exists, it should return a new name with _2
    tested = mk_testfiles(new)  #' "testfile_name_1"
    new = DA.filer.prevent_overwrite(testfile_name, **kws)
    assert (
        new == testfile_name + "_2"
    ), f"new_name = {new}, testfile_name = {tested}"

    # == Cleanup
    dac.cleanfiles(testfile_name)
    for name in distraction_names:
        dac.cleanfiles(name)


if __name__ == "__main__":
    test_prevent_overwrite()
    os.getcwd()
    Path.cwd()

# %%
if __name__ == "__main__":
    import ipytest

    ipytest.run()
