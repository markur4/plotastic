#
# %% imports
import subprocess
import venv # * for creating virtual environments
import os
import requirements as req

# %% [markdown]
# 
# # Setup Environment for this Project
# ### General Strategy: 
# 1. **How to handle virtuel environemts in the first place**
#    1. Do we need pyenv..? Let's try without..
#    2. Just install python versions with `brew`
#    3. Then setup a a virtual environment using `venv`` for each project
# 2. **Setup a virtual environment using `venv`**
#     1. Collect and document requirements in `requirements.py` as you develop
#     2. Run this notebook to setup the environment
#         - Packages defined in `requirements.py` will be installed
#         - requirements.txt file will be exported into project folder for others to use
#         - This project will be added as editable mode to make it testable
# 3. **Install this package with setup.py**
#     - requirements.txt is parsed
#     - package will be installed
# 
# == 
# == 
# == 1. Base Python interpreter =========================================================
# %% Install python
# *  Just use a global one from `brew`!
# !! brew install python@3.11 # !!!! <-- Uncomment

### VSCode will ask us to select a python interpreter: For now, use the global one we installed

# %% Add base python interpreter to `PATH`? 
# - No need to add to `PATH`, since we will use `venv` to create a virtual environment for each project
# - Having out terminal no connection with any python interpreter will make sure that we are not using the wrong interpreter by accident
# !!!! Don't do this, unless required
# !! export PATH="/usr/local/opt/python/libexec/bin:$PATH" 

# %% Check PATH
PATH = subprocess.check_output(["echo $PATH"], shell=True)
PATH_list = PATH.decode("utf-8").split(":")
print(len(PATH_list))
for path in PATH_list:
    print(path)
    

# ==
# ==
# == 2. Setup a Virtual Environment using `venv` =======================================
# %% Describe the folder stucture
# !!!! VScode finds this automatically ONLY if it's in the root folder of the workspace
PROJECT_DIR = "." # !!!! Don't change, or vscode won't find it
ENV_NAME = "venv" # 
ENV_PATH = os.path.join(PROJECT_DIR, ENV_NAME)
ENV_PATH

# %% Create Virtual Environment
# venv.create(env_dir=ENV_PATH, clear=True, with_pip=True) # !!!! <-- Uncomment

# %% [markdown]
# ### Activate Environment
# #### In Terminal:
# - Activate the environment with `source venv/bin/activate` (use `ENV_NAME` instead of `venv`)
#   - Check if the environment is active with `which python`
#   - Should point to the `venv/bin/python` file
# - Deactivate the environment with `deactivate`
# 
# 
# #### In VSCode:
# - Select the environment in VSCode with `Python: Select Interpreter` command (NOT in
#   this notebook!))
#   - Navigate to the `venv/bin/python` file
# - Select environment for THIS NOTEBOOK
#   - VScode will ask you to install the `IPython` extension, do that

# %% Can environment can be activated?
# !!!! This is a notebook, manually select the interpreter for the Jupyter Kernel
! source venv/bin/activate  

# %% Environmant active?
! which python # * -> Should be within the project directory

# == 
# == 
# == 3. Install Project Requirements ===================================================
# %% [markdown]
# ## 3. Install Project Requirements
# 1. **Prepare requirements**
#    1. The *main place* where requirements are gathered and documented is `requirements.py`
#    2. Pip taxes in .txt, use `req.to_txt()` to export requirements to `requirements-dev.txt`
#    3. `requirements-dev.txt` has all packages the developer deems noting
#    4. `requirements.txt` has EVERY package gained from `pip freeze`
# 2. **Install requirements**
#    - Install requirements with `pip install -r requirements-dev.txt`
# 3. **Export requirements** 
#    - `pip freeze > requirements.txt`
#    - This may be overwritten by anyone
# 4. **pip install this Project in editable mode**
#    - `pip install -e .`
#    - Makes it testable

# %% Check Requirements
req.REQUIREMENTS

# %% Check extra requirements that are only installed if needed
req.REQUIREMENTS_EXTRA

# %% Write requirements to file
req.to_txt(fname="requirements-dev.txt", requirements=req.REQUIREMENTS)

# %% If there is already a requirements.txt, use that instead

# %% Install requirements from a file without comments
! pip install -r requirements-dev.txt

# %% Export requirements to be used by the publix
! pip freeze > ../requirements.txt

# == 
# == 
# == 4. Install project in editable mode ===============================================
# %% 
# ? This can take a while
# ! pip install -e . 

# %% Package importable?
import plotastic # !!!! Need to restart kernel if freshly installed

# %% uninstall to reinstall
# ! pip uninstall plotastic -y

# ==
# == 
# == 5. Install Development Tools ======================================================
# %% Check devtools
DEVELOPMENT_TOOLS = req.DEVELOPMENT_TOOLS
DEVELOPMENT_TOOLS

# %% Install devtools
req.to_txt(fname="_devtools.txt", requirements=DEVELOPMENT_TOOLS)
! pip install -r _devtools.txt
os.remove("_devtools.txt") # * cleanup

# ==
# ==
# == Pip slow? =========================================================================
# %% [markdown] 
# ## 6. Is `pip` slow?
# - Check Cache
#   - pip caches wheels and HTML files
#   - without this, pip can be painfully slow (~1-2 min)
#   - Clear cache if needed, to get up to date wheels etc.
# - If there' already a requirements.txt, use `pip install -r requirements.txt`

# %% Find pip cache directory, see if there are things inside
! pip cache dir

#%% Clear cache directory, if needed
# - Retrieve cache directory as python variable
pip_cache_dir = subprocess.check_output(["pip cache dir"], shell=True)
# - Remove cache directory
# !! rm -rf $pip_cache_dir # !!!! <-- Uncomment


# %% Use requirements.txt to install packages if present
if os.path.exists("requirements.txt"):
    # - Install from requirements.txt
    # !! pip install -r requirements.txt # <-- Uncomment
    # - Update Packages





