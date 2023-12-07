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


import DA_configs as dac


# %% testfigure
# import matplotlib.pyplot as plt
# import numpy as np
# fig, ax = plt.subplots(2,2)

# fig.get_axes()
# fig.axes

# %% Test


DA = dac.DA_STATISTICS

funcs = [
    DA.save_statistics,
    # DA.save_fig, # !! Not working, but let's keep it for now
    # DA.save_all, # !! Not working
]


@pytest.mark.parametrize("func", funcs)
def test_save(func: Callable, lastcleanup=True):
    """Test export_statistics()"""

    ### Define a name
    fname = "plotastic_results"

    ### Cleanup before testing
    dac.cleanfiles(fname)

    # == Test overwrite=True ===============================
    kwargs = dict(fname=fname, overwrite=True)
    func(**kwargs)
    func(**kwargs)  #' Should overwrite
    func(**kwargs)  #' Should overwrite

    ### Make sure files overwrote each other
    saved = glob(fname + "*")
    assert len(saved) in [
        1,
        2,
    ], "Should have saved one/two files, insted got: " + str(saved)

    dac.cleanfiles(fname)

    # == Test overwrite="day" ===============================
    kwargs = dict(fname=fname, overwrite="day")
    func(**kwargs)
    func(**kwargs)  #' Should overwrite
    func(**kwargs)  #' Should overwrite

    ### Make sure files didn't delet each other
    saved = glob(fname + "*")
    assert len(saved) in [
        1,
        2,
    ], "Should have saved one or two files, insted got: " + str(saved)

    dac.cleanfiles(fname)

    # == Test overwrite="nothing" ===============================
    kwargs = dict(fname=fname, overwrite="nothing")
    func(**kwargs)
    func(**kwargs)  #' Should NOT overwrite
    func(**kwargs)  #' Should NOT overwrite

    ### Make sure files didn't delete each other
    saved = glob(fname + "*")
    assert len(saved) in [
        3,
        6,
    ], "Should have saved three/six files, insted got: " + str(saved)

    if lastcleanup:
        dac.cleanfiles(fname)


if __name__ == "__main__":
    test_save(func=DA.save_statistics, lastcleanup=False)

    ### cleanup
    # for file in glob("plotastic_results*"):
    #     os.remove(file)

# %%

# %% Test save_fig
# import matplotlib.pyplot as plt
# DA.plot_box_strip()
# DA.save_fig(fname="p1", overwrite=True)  # ? saves wrong fig ?
# DA.save_fig(fname="p2", overwrite=True, fig=DA.fig)  # ? saves wrong fig ??
# DA.fig.savefig("p3.pdf")  # ? saves CORRECT FIG!!
# plt.savefig("p4.pdf")


# %%

# %% interactive testing to display Plots

if __name__ == "__main__":
    import ipytest

    # ipytest.run()
