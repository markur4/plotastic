#
# %% Imports

import matplotlib.pyplot as plt

import plotastic as plst
from plotastic import Annotator

import markurutils as ut

import test_utils as tut


# %% Automatic testing for dataset TIPS

# ! Don't use with empty groups
# ! We chose exclusions that won't show in the plot, but we need those arguments to test it
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


def TIPS(DF, dims, annot_pairwise_kwargs):
    AN = Annotator(data=DF, dims=dims, verbose=True)
    _ph = AN.test_pairwise(paired=False, padjust="none")
    AN = (
        AN.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_pairwise_kwargs,
            show_ph=False,
            only_sig="all",
        )
    )

def test_annotator_tips():
    DF, dims = ut.load_dataset("tips")
    for dim, kwargs in zip(tut.DIMS_TIPS, TIPS_annot_pairwise_kwargs):
        # print("\n !!!", dim)
        # print(" !!!", kwargs)
        TIPS(DF, dim, kwargs)

# %% Automatic Testing for dataset FMRI

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


def FMRI(DF, dims, annot_pairwise_kwargs):
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


def test_annotator_fmri():
    DF, dims = ut.load_dataset("fmri")
    for dim, kwargs in zip(tut.DIMS_FMRI, FMRI_annot_pairwise_kwargs):
        # print("\n !!!", dim)
        # print(" !!!", kwargs)
        FMRI(DF, dim, kwargs)
