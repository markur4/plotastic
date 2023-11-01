"""
- Gather all dependencies you stumble across during development here!
- Document them well
- This is used by 
    - Setup.py
    - Setting up the virtual environment
"""
# %% Imports


# %% List requirements


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
    # "statannot", # * Superseded by statannotations
    "statannotations",
    "pingouin",
    ### Misc
    "joblib",
    "pyperclip",
    "colour",  # * For custom colour maps
    "xlsxwriter",  # * For saving results to excel
    "ipynbname",  # * Used by utils
    "openpyxl",  # * optional for Pandas, but error when not installed
    "icecream",  # * better than print
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

# %% Write to .txt


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


# %% Parsers

# def parse_requirements(
#     fname: str = "requirements-dev.txt", ret_pyversion=False
# ) -> tuple[str, list[str]]:
#     """Parse a requirements.txt file and return a list of requirements.

#     :param filename: Path to requirements.txt file, defaults to "requirements-dev.txt"
#     :type filename: str, optional
#     :param ret_pyversion: Return python version, defaults to False
#     :type ret_pyversion: bool, optional
#     :return: Python version & List of requirements
#     :rtype: list[str]
#     """
#     with open(fname, "r") as f:
#         ### Split into lines and strip whitespace
#         lines = [line.strip() for line in f.readlines()]

#         ### Exclude Lines
#         # fmt: off
#         requirements: list = [
#             line for line in lines
#             # * Exclude empty lines
#             if line
#             # * Exclude comments and strings
#             and not line.startswith(("#", "'", "\"", "-"))
#         ]
#         # fmt: on
#     ### Handle python version
#     # * Retrieve it from requirements
#     python_version: str = requirements.pop(0)

#     # * setup.py uses format: ">=3.7", remove the part before the operator keeping the operator
#     for operator in ["==", ">=", "<=", ">", "<", "!=", "~="]:
#         if operator in python_version:
#             version = python_version.split(operator)[-1]
#             break
#         else:
#             version = None
#     if version is None:
#         raise ValueError(
#             f"Could not find operator {operator} in python version '{python_version}'"
#         )
#     python_version = operator + version

#     if ret_pyversion:
#         return python_version
#     else:
#         return requirements


# if __name__ == "__main__":
#     version = parse_requirements("requirements-dev.txt", ret_pyversion=True)
#     reqs = parse_requirements("requirements-dev.txt")
#     print(version)
#     print(reqs)


# %%
