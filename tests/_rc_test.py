#
# %%
# import pytest
import ipytest
import matplotlib.pyplot as plt

import plotastic as plst

import DA_configs as dac

# %%


def test_rc():
    """Test rc()"""
    plst.set_palette("Set2")
    plst.set_style("paper")

    DA = dac.DA_ALL
    DA.plot_box_strip()

    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()
