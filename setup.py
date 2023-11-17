from setuptools import setup, find_packages

import requirements as req

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

NAME = "plotastic"
PYTHON_VERSION = req.PYTHON_VERSION
REQUIREMENTS = req.REQUIREMENTS
REQUIREMENTS_EXTRA = req.REQUIREMENTS_EXTRA

setup(
    # == Metadata ======================================================
    name=NAME,
    version="0.1.0",
    author="markur4",
    description="Streamlining statistical analysis by using plotting keywords in Python.",
    python_requires=PYTHON_VERSION,
    install_requires=REQUIREMENTS,
    extras_requires=REQUIREMENTS_EXTRA,
    url="https://github.com/markur4/plotastic",
    license="GPLv3",
    # == Package Structure =============================================
    ### Automatically find all packages in src
    # * Those that have an __init__.py AND match the name of the package
    packages=find_packages(where="src", include=[NAME, NAME + ".*"]),
    ### Define location of all packages.
    package_dir={
        "": "src",  # * "" is the package root (where setup.py is)
        # NAME: "src/" + NAME,
        # "dataanalysis": "src/plotastic/dataanalysis",
    },
    # == Non .py Files =================================================
    include_package_data=True # * Include non .py files specified in MANIFEST.in
    ### Required files (e.g. py.typed, documentation...)
    # package_data={
    #     "": [f"py.typed"],  # * "" is the package root (where setup.py is)
    #     "dataanalysis": [f"py.typed"],
    #     "example_data": ["data/*.xlsx"],
    # },
    # ### Extra Files to be installed with the package (e.g. useful .gif .txt, ...)
    # data_files={},
)

### CHECK IF THIS PACKAGE IS INSTALLABLE:
###    $ pip install -e . --dry-run
