# %% Import packages

import markurutils as ut
from plotastic import Annotator


# %% test for FMRI

DF, dims = ut.load_dataset("fmri")
AN = Annotator(
    data=DF,
    dims=dims,
    subject="subject",
    verbose=True,
)

ph = AN.test_pairwise(paired=True, padjust="bonf")
# ut.pp(ph[ph["p-corr"] < 0.0001])


AN, PH2 = (
    AN.subplots()
    .fillaxes(kind="box")
    .annotate_pairwise(
        include="__hue",  # * This will only annotate those pairs comparing different hues!
        # include=[0, "stim"],
        # exclude=[1, "cue", {1: ("cue", "stim")}],
        # exclude=[1, {"stim": (0, 2)}],
        # exclude="__X",
        # exclude=[1, "cue", {"cue": ("cue", "stim")}], # ! Correct error
        # include_in_facet={"frontal": [0, "cue"], (0,1): [0, "cue"]}, # ! Correct error
        # include_in_facet={"frontal": [0, "cue"], "parietal": [0, "cue"]},
        # exclude_in_facet={"frontal": [2, "cue"], "parietal": [4, "stim"]},
        # include_in_facet={
        #     "frontal": [0, "cue", {"stim": (3, 4)}],
        #     "parietal": [0, "cue", {"stim": (4, 6)}],
        # },
        # exclude_in_facet={
        #     "frontal": [2, "cue", {"stim": (3, 7)}],
        #     "parietal": [4, "stim", {"stim": (2, 9)}],
        # },
        return_ph=True,  # * Returns posthoc table after selection arguments were applied
    )
)

ut.pp(PH2[PH2["p-corr"] < 0.00001])


# %% Test for tips

DF, dims = ut.load_dataset("tips")
AN2 = Annotator(
    data=DF,
    dims=dims,
    verbose=True,
)

ph = AN2.test_pairwise(paired=False, padjust="none")

AN2 = (
    AN2.subplots()
    .fillaxes(kind="box")
    .annotate_pairwise(
        # include=["Yes", {"1-2": ("Yes", "No")}],
        # exclude=["No", {"Yes": ("1-2", ">=3")}],
        # include_in_facet={
        #     ("Lunch", "Male"): ["Yes", {">=3": ("Yes", "No")}],
        #     ("Lunch", "Female"): ["No", {"No": ("1-2", ">=3")}],
        # },
        # exclude_in_facet={
        #     ("Lunch", "Male"): ["Yes", {">=3": ("No", "Yes")}],
        #     ("Lunch", "Female"): ["No", {"Yes": ("1-2", ">=3")}],
        # },
        only_sig="tolerant",
        # show_ph=True,
    )
)


# %% Automatic testing for dataset TIPS

# ! PostHoc does not support dimensions that produce empty groups in dataframe
TIPS_dimses = [
    # dict(y="tip", x="day", hue="sex", col="smoker", row="time"), # ! these make empty groups
    dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
    dict(y="tip", x="size-cut", hue="smoker", col="sex"),
    dict(y="tip", x="size-cut", hue="smoker"),
    dict(y="tip", x="size-cut"),
]

# ! We chose exclusions that will never annotate anything, but we need those arguments to test it
TIPS_annot_pairwise_kwargs = [
    dict(
        include=["Yes", {"1-2": ("Yes", "No")}],
        exclude=["No", {"Yes": ("1-2", ">=3")}],
        include_in_facet={
            ("Lunch", "Male"): ["Yes", {">=3": ("Yes", "No")}],
            ("Lunch", "Female"): ["No", {"No": ("1-2", ">=3")}],
        },
        exclude_in_facet={
            ("Lunch", "Male"): ["Yes", {">=3": ("No", "Yes")}],
            ("Lunch", "Female"): ["No", {"Yes": ("1-2", ">=3")}],
        },
    ),
    dict(
        include=["Yes", {"1-2": ("Yes", "No")}],
        exclude=["No", {"Yes": ("1-2", ">=3")}],
        include_in_facet={
            "Male": ["Yes", {">=3": ("Yes", "No")}],
            "Female": ["No", {"No": ("1-2", ">=3")}],
        },
        exclude_in_facet={
            "Male": ["Yes", {">=3": ("No", "Yes")}],
            "Female": ["No", {"Yes": ("1-2", ">=3")}],
        },
    ),
    dict(
        include=["Yes", {"1-2": ("Yes", "No")}],
        exclude=["No", {"Yes": ("1-2", ">=3")}],
    ),
    dict(
        include=["1-2"],
        exclude=[">=3"],
    ),
]


def TIPS_tester(DF, dims, annot_pairwise_kwargs):
    AN = Annotator(data=DF, dims=dims, verbose=True)
    ph = AN.test_pairwise(paired=False, padjust="none")
    AN = (
        AN.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_pairwise_kwargs,
            show_ph=False,
            only_sig="all",
        )
    )


DF, dims = ut.load_dataset("tips")
for dim, kwargs in zip(TIPS_dimses, TIPS_annot_pairwise_kwargs):
    print("\n !!!", dim)
    print(" !!!", kwargs)
    TIPS_tester(DF, dim, kwargs)


# %% Automatic Testing for dataset FMRI

FMRI_dimses = [
    dict(y="signal", x="timepoint", hue="event", col="region"),
    dict(y="signal", x="timepoint", hue="region", col="event"),
    dict(y="signal", x="timepoint", hue="region"),
    dict(y="signal", x="timepoint", hue="event"),
    dict(y="signal", x="timepoint"),
]

FMRI_annot_pairwise_kwargs = [
    dict(
        include=[0, "stim"],
        exclude=[1, {"stim": (0, 2)}],
        include_in_facet={
            "frontal": [0, "cue", {"stim": (3, 4)}],
            "parietal": [0, "cue", {"stim": (4, 6)}],
        },
        exclude_in_facet={
            "frontal": [2, "cue", {"stim": (3, 7)}],
            "parietal": [4, "stim", {"stim": (2, 9)}],
        },
    ),
    dict(
        include=[0, "frontal"],
        exclude=[1, {"frontal": (0, 2)}],
        include_in_facet={
            "stim": [0, "frontal", {"parietal": (3, 4)}],
            "cue": [0, "parietal", {"frontal": (4, 6)}],
        },
        exclude_in_facet={
            "stim": [2, "parietal", {"frontal": (3, 7)}],
            "cue": [4, "frontal", {"parietal": (2, 9)}],
        },
    ),
    dict(
        include=[0, "frontal"],
        exclude=[1, {"frontal": (0, 2)}],
    ),
    dict(
        include=[0, "cue"],
        exclude=[1, {"stim": (0, 2)}],
    ),
    dict(
        include=[0, 2],
        exclude=[
            1,
        ],
    ),
]


def FMRI_tester(DF, dims, annot_pairwise_kwargs):
    AN = Annotator(data=DF, dims=dims, verbose=True, subject="subject")
    ph = AN.test_pairwise(paired=True, padjust="bonf")
    AN = (
        AN.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_pairwise_kwargs,
            show_ph=False,
            only_sig="strict",
        )
    )


DF, dims = ut.load_dataset("fmri")
for dim, kwargs in zip(FMRI_dimses, FMRI_annot_pairwise_kwargs):
    print("\n !!!", dim)
    print(" !!!", kwargs)
    TIPS_tester(DF, dim, kwargs)
