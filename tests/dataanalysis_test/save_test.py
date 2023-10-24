#
# %% imports
from typing import Callable

import os
from glob import glob

import pytest

# import seaborn as sns
# import pandas as pd

# import markurutils as ut
import plotastic as plst

import conftest as ct


#%% testfigure
# import matplotlib.pyplot as plt
# import numpy as np
# fig, ax = plt.subplots(2,2)

# fig.get_axes()
# fig.axes

# %% Test


DA = ct.DA_COMPLETE_STATISTICS
funcs = [
    DA.save_statistics,
    DA.save_fig,
    DA.save_all,
]


@pytest.mark.parametrize("func", funcs)
def test_save(func: Callable, lastcleanup=True):
    """Test export_statistics()"""

    ### Define a name
    fname = "plotastic_results"

    ### Cleanup before testing
    ct.cleanfiles(fname)

    # == Test overwrite=True ===============================
    kwargs = dict(fname=fname, overwrite=True)
    func(**kwargs)
    func(**kwargs)  # * Should overwrite
    func(**kwargs)  # * Should overwrite

    ### Make sure files overwrote each other
    saved = glob(fname + "*")
    assert len(saved) in [1, 2], "Should have saved one/two files, insted got: " + str(
        saved
    )

    ct.cleanfiles(fname)


    # == Test overwrite="day" ===============================
    kwargs = dict(fname=fname, overwrite="day")
    func(**kwargs)
    func(**kwargs)  # * Should overwrite
    func(**kwargs)  # * Should overwrite

    ### Make sure files didn't delet each other
    saved = glob(fname + "*")
    assert len(saved) in [
        1,
        2,
    ], "Should have saved one or two files, insted got: " + str(saved)

    ct.cleanfiles(fname)
    
    # == Test overwrite="nothing" ===============================
    kwargs = dict(fname=fname, overwrite="nothing")
    func(**kwargs)
    func(**kwargs)  # * Should NOT overwrite
    func(**kwargs)  # * Should NOT overwrite

    ### Make sure files didn't delet each other
    saved = glob(fname + "*")
    assert len(saved) in [
        3,
        6,
    ], "Should have saved three/six files, insted got: " + str(saved)

    if lastcleanup:
        ct.cleanfiles(fname)


if __name__ == "__main__":
    test_save(func=DA.save_all, lastcleanup=False)

    # os.remove(out + ".xlsx")




# %% interactive testing to display Plots

if __name__ == "__main__":
    import ipytest

    ipytest.run()
