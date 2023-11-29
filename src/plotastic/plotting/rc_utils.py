"""Utilities for setting rcParams and styles"""

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# import markurutils as ut
import plotastic.utils.utils as ut
from plotastic.dataanalysis.dataanalysis import DataAnalysis
from plotastic.example_data.load_dataset import load_dataset

import plotastic.plotting.rc as rc


# %%
def print_styles() -> str:
    print("\n".join([f"{k}:\n\t{v}" for k, v in rc.STYLENAMES.items()]))


def set_rcParams(rcParams: dict):
    """Iterates through settings dictionary and applies them to
    matplotlib rcParams via mpl.rcParams [setting] = value.

    :param rcParams: _description_
    :type rcParams: dict
    """
    for setting, value in rcParams.items():
        mpl.rcParams[setting] = value


def set_style(style: dict | str) -> None:
    """Checks if style is set by plotastic, if not checks if style is a
    dict with rcParams as keys and values, if not checks if style is a
    matplotlib style and mpl.style.use(style), if not uses seaborn styleplott

    :param style: _description_
    :type style: dict | str
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """

    ### Set matplotlib settings
    if style in rc.STYLENAMES["plotastic"]:
        set_rcParams(rc.STYLES[style])
    elif isinstance(style, dict):
        set_rcParams(style)
    elif style in mpl.style.available:
        mpl.style.use(style)
    else:
        try:
            sns.set_style(style)
        except ValueError:
            m = [
                f"#! Style '{style}' not found. Choose one",
                f"from these: {print_styles()}",
            ]
            raise ValueError(" ".join(m))


# %%
def set_palette(palette: str | list = "Paired", verbose=True):
    """Sets the color palette.

    :param palette: _description_, defaults to "Paired"
    :type palette: str | list, optional
    :param verbose: _description_, defaults to True
    :type verbose: bool, optional
    """
    if verbose:
        pal = sns.color_palette(palette, 8).as_hex()
        print(f"#! You chose this color palette: {pal}")
        if ut.is_notebook():
            from IPython.display import display

            display(pal)

    # sns.set_theme(palette=palette) # !! resets rcParams
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=sns.color_palette(palette)
    )
