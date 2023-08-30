from setuptools import setup, find_packages


# from markurutils import __name__, __download_url__, __version__, __author__, install_requires
# from markurutils.__init__ import name, url, __version__, __author__, requirements


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

setup(
    description="A wrapper for seaborn plotters for convenient statistics powered by pingouin!",
    python_requires=">=3.10",  # * 3.11 recommended, 3.10 might work, too
    name=NAME,
    version="0.0.1",
    url="https://github.com/markur4/plotastic",
    author="markur4",
    # * List all packages. If you have just one package, put the name of the package.
    packages=[NAME],
    # * Package paths relative to setup.py that each have a __init__.py file
    package_dir={NAME: "plotastic"},
    # * Non- .py files (e.g. py.typed, documentation...) required by package
    package_data={NAME: ["plotastic/py.typed"]},
    # * Non- .py files (e.g. .gif .txt, ...) that should be installed with the package
    data_files={},
    install_requires=REQUIREMENTS,
    license="MIT",
)

### CHECK IF THIS PACKAGE IS INSTALLABLE:
###    $ pip install -e . --dry-run
