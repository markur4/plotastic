#
# %%
import venv

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
! pip install pytest pytest-cov 

# %%
! pytest --cov=src/plotastic --cov-report=xml

# %%
