import pyperclip


def print_imports(to_clipb=True, include='all') -> None:
    s = '''
# from __future__ import annotations # for type hinting my Class type for return values

### <<< ESSENTIALS >>>
import markurutils as ut
from markurutils import pp


import numpy as np
import pingouin as pg
import pandas as pd 
pd.set_option("display.precision", 3)
def catchstate(df, var_name:str ="df") -> pd.DataFrame:
    """Helper function that captures intermediate Dataframes.
    THIS FUNCTIONS DOESN'T WORK WHEN IMPORTED FROM MODULE
    In the global namespace, make a new variable called var_name and set it to dataframe
    :param df: Pandas dataframe
    :param var_name:
    :return:
    """
    globals()[var_name] = df
    return df 
df = pd.DataFrame() # * Avoid linting errors

from pathlib import Path
import pyrectories as prct
prct.set_option(overwrite=True,
                prefix="_", use_subsubfolder=True, use_daytime=True,
                )
import plotastic as plst
FONTSIZE = 10
plst.set_seaborn_style(context="paper",
                       palette=['#1f78b4', '#a6cee3', '#33a02c', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00'],
                       show_palette=True,

                       rc_theme_kws={'legend.markerscale': 1,
                                     'font.weight': 'normal',
                                     'axes.labelweight': 'normal',
                                     'axes.titleweight': 'normal',
                                     'axes.labelsize': FONTSIZE,  # fontsize of the x any y labels
                                     'font.size': FONTSIZE,
                                     'xtick.labelsize': FONTSIZE - 1,
                                     'ytick.labelsize': FONTSIZE - 1,
                                     'legend.title_fontsize': FONTSIZE,
                                     'legend.fontsize': FONTSIZE,
                                     },
                       rc_style_kws={'font.family': 'sans-serif',
                                     'font.sans-serif': 'Arial', 'axes.grid': True,
                                     'font.weight': 'bold',
                                     }
                       )
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

# import pyensembl
# GENOME = pyensembl.EnsemblRelease(species="human")
'''
    complete = '''
### <<< BUILTIN TYPES >>>
# import re
# import inspect
# import itertools
# from copy import copy, deepcopy
# from functools import wraps
# from typing import Callable, ClassVar, List, Dict, Iterable

### <<< TIME >>>
# import time
# from datetime import datetime, date

### <<< I/O >>>
# from IPython.display import display, HTML
# import ipynbname
# from pathlib import Path
# import os
# from contextlib import contextmanager
# import warnings
# import weasyprint # PDF generation from DataFrames

### <<< TABLES >>>
# import pandas_profiling as pdp
# import scipy.stats
# from statsmodels.stats import anova
# from statannot import add_stat_annotation

### <<< MPL >>>
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
# print("importing matplotlib version: ", mpl.__version__)
# from matplotlib import pyplot as plt
# from matplotlib.colors import to_hex
# from matplotlib.ticker import PercentFormatter
# import matplotlib.font_manager as font_manager

### <<< OOP >>>
# from dataclasses import dataclass, field

    '''

    if complete == 'all':
        s = s+complete

    if to_clipb: ### QUICKLY ctrl-V !
        pyperclip.copy(s)

    print(s)