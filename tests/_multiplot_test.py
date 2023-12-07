#
# %%

import pytest
import ipytest
import matplotlib.pyplot as plt

import plotastic as plst

import DA_configs as dac


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_plot(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot()
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_box_strip(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot_box_strip()
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()


# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def plot_box_swarm(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    DA.plot_box_strip()
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ipytest.run()
