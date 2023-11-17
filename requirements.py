#
#%% [markdown]
# == What's this file? =================================================

# - Gather all dependencies you stumble across during development here!
# - Document them well
# - This is used by 
#     - Setup.py
#     - Setting up the virtual environment

# %% 
### Imports


# %% 
# == List requirements =================================================

# fmt: off

PYTHON_VERSION = ">=3.11"  # * Used only by setup.py
REQUIREMENTS = [
    ### Core
    "numpy",
    "pandas==1.5.3",  # !! pingouin Not working with pandas 2.0 yet
    ### Plotting
    "matplotlib",
    "seaborn",
    ### Statistics
    "scipy",
    # "statannot",   # * Superseded by statannotations
    "statannotations",
    "pingouin",
    ### Misc
    "joblib",
    "pyperclip",
    "colour",       # * For custom colour maps
    "xlsxwriter",   # * For saving results to excel
    "ipynbname",    # * Used by utils
    "openpyxl",     # * Optional for Pandas, but error when not installed
    "icecream",     # * Better than print
    # "tabulate",   # * For printing tables, use .to_markdown() instead
]
REQUIREMENTS_EXTRA = [  # ? Only installed if needed, whatever that means
    #
]
DEVELOPMENT_TOOLS = [
    "pytest",
    "ipytest",
    "pytest-cov", # * Displays how much of code was covered by testing
    "nbconvert", # * For converting notebooks to markdown
]

# fmt: on

# %% 
# == Write requirements to file ========================================


def to_txt(fname: str = "requirements-dev.txt", requirements: list[str] = None) -> None:
    """Write a requirements.txt file from a list of requirements.

    :param filename: Path to requirements.txt file, defaults to "requirements-dev.txt"
    :type filename: str, optional
    :param requirements: List of requirements in this format "package==version", defaults to None
    :type requirements: list[str], optional
    """
    with open(fname, "w") as f:
        ### Write requirements
        for requirement in requirements:
            f.write(f"{requirement}\n")

