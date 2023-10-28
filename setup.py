from setuptools import setup, find_packages

import plotastic.requirements as req

# requirements_file = "_setup_env/requirements-dev.txt"

NAME = "plotastic"
# PYTHON_VERSION = eu.parse_requirements(fname=requirements_file, ret_pyversion=True)
# REQUIREMENTS = eu.parse_requirements(fname=requirements_file)
PYTHON_VERSION = req.PYTHON_VERSION
REQUIREMENTS = req.REQUIREMENTS
REQUIREMENTS_EXTRA = req.REQUIREMENTS_EXTRA

# https://setuptools.pypa.io/en/latest/references/keywords.html
### LAYOUT:
# * project_root/                 # Project root: 'plotastic'
# * ├── setup.py
# * ├── ...
# * └── src/                      # Source root
# *    └── package/               # Package root: 'plotastic'
# *        ├── __init__.py
# *        ├── py.typed
# *        ├── ...
# *        ├── (module.py)
# *        ├── subpkg1/           # Subpackage root: 'plotastic.dimensions'
# *        │   ├── __init__.py
# *        │   ├── ...
# *        │   └── module1.py
# *        └── subpkg2/           # Subpackage root: 'plotastic.plotting'
# *            ├── __init__.py
# *            ├── ...
# *            └── module2.py

setup(
    # == Metadata ======================================================================
    name=NAME,
    version="0.0.1",
    author="markur4",
    description="A wrapper for seaborn plotters for convenient statistics powered by pingouin!",
    python_requires=PYTHON_VERSION,
    install_requires=REQUIREMENTS,
    extras_requires=REQUIREMENTS_EXTRA,
    url="https://github.com/markur4/plotastic",
    license="MIT",
    # == Package Structure =============================================================
    ### Automatically find all packages in src
    # * Those that have an __init__.py AND match the name of the package
    packages=find_packages(where="src", include=[NAME]),
    ### Define location of all packages.
    package_dir={
        "": "src",  # * "" is the package root (where setup.py is)
        # NAME: f"src/{NAME}",
        # "dataanalysis": "src/plotastic/dataanalysis",
    },
    # == Non .py Files =================================================================
    ### Required files (e.g. py.typed, documentation...)
    package_data={
        "": [f"py.typed"],  # * "" is the package root (where setup.py is)
        "dataanalysis": [f"py.typed"],
        "example_data": ["data/*.xlsx"],
    },
    ### Extra Files to be installed with the package (e.g. useful .gif .txt, ...)
    data_files={},
)

### CHECK IF THIS PACKAGE IS INSTALLABLE:
###    $ pip install -e . --dry-run
