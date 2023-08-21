#
#%% Imports
import markurutils as ut
import plotastic as plst
from plotastic.posthoc import PostHoc








# %% test for dataset fmri

DF, dims = ut.load_dataset("fmri")
PH = PostHoc(
    data=DF,
    dims=dims,
    subject="subject",
    verbose=True,
)
# DA.data_describe()
PH.catplot()

PH.test_pairwise(
    paired=True,
    padjust="bonf",
    subject="subject",
    return_desc=False,
)


# %% Experiment with between within or mixed design

PH.test_pairwise(
    # between=[dims["x"], dims["hue"]],
    # within=[dims["x"], dims["hue"]],
    within=["ddd", dims["hue"]],
    # between=dims["col"],
    subject="subject",
    parametric=True,
    padjust="bh",
    nan_policy="pairwise",
)

# %%  test for dataset tips

DF, dims = ut.load_dataset("tips")
PH = PostHoc(
    data=DF,
    dims=dims,
    subject="day",
    verbose=True,
)

PH.catplot()

PH.test_pairwise(
    paired=False,  # ! If using paired, but table is not paired, it will throw a cryptic error "AssertionError: The T-value must be a int or a float."
    padjust="bonf",
    # subject="subject",
    return_desc=False,
)




# %%Automatic testing

# ! PostHoc does not support dimensions that produce empty groups in dataframe
dimses_tips = [
    # dict(y="tip", x="day", hue="sex", col="smoker", row="time"), # ! these make empty groups
    # dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    # dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
    dict(y="tip", x="size-cut", hue="smoker", col="sex"),
    dict(y="tip", x="size-cut", hue="smoker"),
    dict(y="tip", x="size-cut"),
]


dimses_fmri = [
    dict(y="signal", x="timepoint", hue="event", col="region"),
    dict(y="signal", x="timepoint", hue="region", col="event"),
    dict(y="signal", x="timepoint", hue="region"),
    dict(y="signal", x="timepoint", hue="event"),
    dict(y="signal", x="timepoint"),
]


def tester_tips(DF, dims):
    DA = PostHoc(data=DF, dims=dims, verbose=True)
    # DA.catplot()
    DA.test_pairwise(paired=False)


def tester_fmri(DF, dims):
    DA = PostHoc(data=DF, dims=dims, verbose=True, subject="subject")
    DA.test_pairwise(paired=True)


DF, dims = ut.load_dataset("tips")
for dim in dimses_tips:
    print("\n !!!", dim)
    tester_tips(DF, dim)

DF, dims = ut.load_dataset("fmri")
for dim in dimses_fmri:
    print("\n !!!", dim)
    tester_tips(DF, dim)