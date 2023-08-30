from setuptools import setup, find_packages


NAME = "plotastic"
REQUIREMENTS = (
    "seaborn",
    "matplotlib",
    "numpy",
    "pandas==1.5.3",  # ! pingouin Not working with 2.0 yet
    "scipy",
    # "statannot",
    "statannotations",
    "pingouin",
    "pyperclip",
    # "statsmodels",
    # "joblib",
    # "ipynbname",
)

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
    description="A wrapper for seaborn plotters for convenient statistics powered by pingouin!",
    python_requires=">=3.10",  # * 3.11 recommended, 3.10 might work, too
    name=NAME,
    version="0.0.1",
    url="https://github.com/markur4/plotastic",
    author="markur4",
    # * Find all packages in src that have a __init__.py file and match the name of the package
    packages=find_packages(where="src", include=[NAME]),
    # * Define location of all packages. "" is the current directory (where setup.py is)
    package_dir={"": "src"},
    # * Non- .py files (e.g. py.typed, documentation...) required by package
    package_data={NAME: [f"src/{NAME}/py.typed"]},
    # * Non- .py files (e.g. .gif .txt, ...) that should be installed with the package
    data_files={},
    install_requires=REQUIREMENTS,
    license="MIT",
)

### CHECK IF THIS PACKAGE IS INSTALLABLE:
###    $ pip install -e . --dry-run
