#
# %% Imports

import pytest

import plotastic as plst
from plotastic.example_data.load_dataset import FILES

# %%

parameters = [name for name in FILES.keys()]


@pytest.mark.parametrize("name", parameters)
def test_load_dataset(name: str):
    """simply checks, if it's executable, after correct packaging in setup.py and all."""
    df, dims = plst.load_dataset(name, verbose=True)


# %%

if __name__ == "__main__":
    import ipytest
    ipytest.run()
