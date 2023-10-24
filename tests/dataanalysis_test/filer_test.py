#
# %% imports

import pytest

import pandas as pd

import os
from glob import glob
from pathlib import Path

import plotastic as plst

import conftest as ct


# %%
import IPython

IPython.extract_module_locals()[1].get("__vsc_ipynb_file__")


# %% Test

DF, dims = plst.load_dataset("tips", verbose=False)
DA = plst.DataAnalysis(DF, dims, verbose=False)
DA_FULL = ct.get_DA_with_full_statistics()


# %% Test prevent_overwrite


def test_prevent_overwrite():
    ### Define a name
    testfile_name = "TEST_FILE_123"

    def make_testfiles(testfile_name):
        ### Make a testfile excel
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_excel(testfile_name + ".xlsx")

        ### Make a testfile text as a distraction
        with open(testfile_name + ".txt", "w") as f:
            f.write("I am an text file")

        return testfile_name

    ### Make sure it's clean before testing
    testfiles = glob(testfile_name + "*")
    for file in testfiles:
        os.remove(file)

    # == TEST

    ### If NO file exists, it should return the same name with _0
    new = DA.filer.prevent_overwrite(testfile_name, ret_parent=False)
    assert new == testfile_name + "_0", f"new_name = {new}"

    ### If a file EXISTS, it should return the same name with _0
    tested = make_testfiles(testfile_name)
    new = DA.filer.prevent_overwrite(testfile_name, ret_parent=False)
    assert new == testfile_name + "_0", f"new_name = {new}, testfile_name = {tested}"

    ### If a file with _0 exists, it should return a new name with _1
    tested = make_testfiles(testfile_name + "_0")
    new = DA.filer.prevent_overwrite(testfile_name, ret_parent=False)
    assert new == testfile_name + "_1", f"new_name = {new}, testfile_name = {tested}"

    ### If a file with _1 exists, it should return a new name with _2
    tested = make_testfiles(testfile_name + "_1")
    new = DA.filer.prevent_overwrite(testfile_name, ret_parent=False)
    assert new == testfile_name + "_2", f"new_name = {new}, testfile_name = {tested}"

    # == Cleanup
    testfiles = glob(testfile_name + "*")
    for file in testfiles:
        os.remove(file)


if __name__ == "__main__":
    test_prevent_overwrite()
    os.getcwd()
    Path.cwd()

# %%
if __name__ == "__main__":
    import ipytest

    # ipytest.run()
