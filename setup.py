from setuptools import setup, find_packages


# from markurutils import __name__, __download_url__, __version__, __author__, install_requires
# from markurutils.__init__ import name, url, __version__, __author__, requirements


NAME = "plotastic"
REQUIREMENTS = (
    "seaborn",
    "matplotlib",
    "numpy",
    "pandas==1.5.3", # ! pingouin Not working with 2.0 yet
    "scipy",
    # "statannot",
    "statannotations",
    "pingouin",
    "pyperclip",
    # "statsmodels",
    # "joblib", "pyperclip", "ipynbname",
)


setup(
    description="A wrapper for seaborn plotters for convenient statistics powered by pingouin!",
    name=NAME,
    version="0.0.1",
    url="https://github.com/markur4/plotastic",
    author="markur4",
    packages=[NAME],
    install_requires=REQUIREMENTS,
    # package_data={
    #    'sample': ['sample_data.csv'],
    license="MIT",
)

### CHECK IF THIS PACKAGE IS INSTALLABLE:
###    $ pip install -e . --dry-run
