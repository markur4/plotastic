# %%

import pytest
import ipytest
import matplotlib.pyplot as plt

import plotastic as plst

import _DA_configs as dac 

# %%

# %%
@pytest.mark.parametrize("DF, dims", dac.zipped_ALL)
def test_edit_legend(DF, dims):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
    if DA.dims.hue:
        DA.plot().edit_legend()
    
    if __name__ == "__main__":
        plt.show()
    else:
        plt.close("all")
    
if __name__ == "__main__":
    ipytest.run()