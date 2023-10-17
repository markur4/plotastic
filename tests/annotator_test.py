#
# %% Imports

import matplotlib.pyplot as plt
import pytest
import ipytest

import plotastic as plst
from plotastic import Annotator

import markurutils as ut

import conftest as ct


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

### Add a column of args: (DF, dims) -> (DF, dims, kwargs)
zipped_tips: list[tuple] = ct.add_zip_column(
    ct.zipped_noempty_tips, TIPS_annot_pairwise_kwargs
)


@pytest.mark.parametrize("DF, dims, annot_kwargs", zipped_tips)
def test_pairwiseannotations_tips(DF, dims, annot_kwargs):
    AN = Annotator(data=DF, dims=dims, verbose=True)
    _ph = AN.test_pairwise(paired=False, padjust="none")
    AN = (
        AN.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_kwargs,
            show_ph=False,
            only_sig="all",
        )
    )


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
        exclude=[1],
    ),
]

### Add a column of args: (DF, dims) -> (DF, dims, kwargs)
zipped_fmri: list[tuple] = ct.add_zip_column(
    ct.zipped_noempty_fmri, FMRI_annot_pairwise_kwargs
)


@pytest.mark.parametrize("DF, dims, annot_kwargs", zipped_fmri)
def test_pairwiseannotations_fmri(DF, dims, annot_kwargs):
    AN = Annotator(data=DF, dims=dims, verbose=True, subject="subject")
    _ph = AN.test_pairwise(paired=True, padjust="bonf")
    AN = (
        AN.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_kwargs,
            show_ph=False,
            only_sig="strict",
        )
    )


# %% interactive testing to display Plots

if __name__ == "__main__":
    ipytest.run()
