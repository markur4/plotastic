#
# %%
from pathlib import Path
import shutil
import venv
import subprocess as sp


# %%
### Define paths
def makepath(*args) -> str:
    return str(Path(*args))

# :: Check Project Root!
ROOT = ".."  

PROJ = makepath(ROOT, ".")  # > Project location for editable install
VENV = makepath(ROOT, "venv")  # > Virtual environment location
PYTHON = makepath(ROOT, "venv", "bin", "python")  # > Python executable
REQUIREMENTS = makepath(ROOT, "requirements.txt")


# %%
### make virtual environment
# > Delete venv if it exists
if Path(VENV).exists():
    shutil.rmtree(VENV)

venv.create(VENV, with_pip=True)


# %%
### Install this project
sp.run([PYTHON, "-m", "pip", "install", "-e", PROJ])
# %%
### Create requirements.txt
with open(REQUIREMENTS, "w") as f:
    sp.call(
        [
            PYTHON,
            "-m",
            "pip",
            "freeze",
            "--exclude-editable",
            "-l",
            ">",
            REQUIREMENTS,
        ],
        stdout=f,
    )
# %%
### Install devtools
sp.run([PYTHON, "-m", "pip", "install", "-e", f"{PROJ}[dev]"])


# %%
#:: Switch to venv !! ==================================================
# %%
### test packages
import numpy as np

np.__version__

# %%
import pytest

pytest.__version__


# %% 
### Make a user venv
import venv
venv.create("venv_user", with_pip=True)