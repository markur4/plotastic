#
# %% imports

import pytest

from pathlib import Path

import plotastic as plst

import conftest as ct


# %%
import IPython
IPython.extract_module_locals()[1].get('__vsc_ipynb_file__')


# %% Test

DF, dims = plst.load_dataset("tips", verbose=False)
DA = plst.DataAnalysis(DF, dims, verbose=False)


# %% Is Notebook
def test_is_notebook():
    shell = DA.filer._current_shell
    is_notebook: bool = DA.filer._script_is_notebook
    print(is_notebook)
    
if __name__ == "__main__":
    test_is_notebook()


#%% Script Name

def test_script_name():
    """Test _get_current_filename()"""
    filename = DA.filer._script_name
    print(filename)

if __name__ == "__main__":
    test_script_name()
    bla = exec("current_script = Path(__file__).stem")
    current_script
    bla = eval("Path(__file__).stem")
    # __file__
    
    # import sys
    # from pathlib import Path
    # Path(sys.argv[0]).stem

# %%
if __name__ == "__main__":
    import ipytest

    # ipytest.run()
