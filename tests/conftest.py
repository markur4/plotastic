"""Utilities for testing plotastic. Contains lists of arguments"""

# %% imports

import pandas as pd

import markurutils as ut


# %% Datasets

### Source of files is seaborn, markurutils just adds cut column
# DF_fmri, dims_fmri = ut.load_dataset("fmri")
# DF_fmri.to_excel("dataframes/fmri.xlsx")  

# DF_tips, dims_tips = ut.load_dataset("tips")
# DF_tips.to_excel("dataframes/tips.xlsx")  

### Load from files
DF_fmri = pd.read_excel("dataframes/fmri.xlsx")
DF_tips = pd.read_excel("dataframes/tips.xlsx")  


# %% Arguments

### Args that make empty groups
dims_withempty_tips = [
    dict(y="tip", x="day", hue="sex", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="sex", hue="day", col="time"),
    dict(y="tip", x="sex", hue="day", row="time"),
    dict(y="tip", x="sex", hue="day", row="size-cut"),
    dict(y="tip", x="sex", hue="day"),
    dict(y="tip", x="sex"),
    dict(y="tip", x="size-cut"),
]

### Args making sure they don't make empty groups
# ! Don't add more, other tests assume 4 entries
dims_noempty_tips = [
    dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
    dict(y="tip", x="size-cut", hue="smoker", col="sex"),
    dict(y="tip", x="size-cut", hue="smoker"),
    dict(y="tip", x="size-cut"),
]

dims_noempty_fmri = [
    dict(y="signal", x="timepoint", hue="event", col="region"),
    dict(y="signal", x="timepoint", hue="region", col="event"),
    dict(y="signal", x="timepoint", hue="region"),
    dict(y="signal", x="timepoint", hue="event"),
    dict(y="signal", x="timepoint"),
]

#%%  Combine for pytest.parametrize

zipped_noempty_tips =  [(DF_tips, dim) for dim in dims_noempty_tips]
zipped_noempty_fmri =  [(DF_fmri, dim) for dim in dims_noempty_fmri]
zipped_noempty_all = zipped_noempty_tips + zipped_noempty_fmri



# %%


# def noempty(func: Callable, **kwargs):
#     """Repeatedly calls function with different arguments. Arguments don't split data
#     into empty groups.


#     :param func: Function that takes a DataFrame and a dict of dimensions as arguments
#     :type func: Callable
#     """

#     for dim in dims_noempty_tips:
#         func(DF_tips, dim, **kwargs)

#     for dim in dims_noempty_fmri:
#         func(DF_fmri, dim, **kwargs)
