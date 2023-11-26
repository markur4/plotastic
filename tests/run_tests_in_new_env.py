"""A helper script to execute tests in a new virtual environment. Not
needed if tomltovenv is used to create the virtual environment."""
#
# %%
import os
import shutil
import venv

#%% 
### Delete environment if present
if os.path.exists("../venv_not_e"):
    shutil.rmtree("../venv_not_e")

# %%
### Create virtual environment
# !! we're inside the tests folder
venv.create(env_dir="../venv_not_e", clear=True, with_pip=True)

#%%
! source venv_not_e/bin/activate

#%% 
### Install non editable for testing
! pip install -r requirements.txt
! pip install git+https://github.com/markur4/plotastic.git
! pip install pytest pytest-cov ipytest

# %%
# !! Coverage requires editable mode
! pytest

# # ! pytest --cov --cov-report=xml

# %%
