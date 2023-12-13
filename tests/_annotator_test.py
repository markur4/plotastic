#
# %% Imports

import matplotlib.pyplot as plt
import pytest

import plotastic as plst
# from plotastic.dataanalysis.annotator import Annotator


import DA_configs as dac


# %% testing for dataset TIPS

# !! Don't use with empty groups
# !! We chose exclusions that won't show in the plot, but we need those arguments to test it
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
zipped_tips: list[tuple] = dac.add_zip_column(
    dac.zipped_noempty_tips, TIPS_annot_pairwise_kwargs
)


@pytest.mark.parametrize("DF, dims, annot_kwargs", zipped_tips)
def test_pairwiseannotations_tips(DF, dims, annot_kwargs):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=True)
    _ph = DA.test_pairwise(paired=False, padjust="none")
    DA = (
        DA.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_kwargs,
            show_ph=False,
            only_sig="all",
        )
    )
    ### Don't plot while executing pytest in terminal
    if __name__ != "__main__":
        plt.close()


# %% Testing for dataset FMRI

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
zipped_fmri: list[tuple] = dac.add_zip_column(
    dac.zipped_noempty_fmri, FMRI_annot_pairwise_kwargs
)


@pytest.mark.parametrize("DF, dims, annot_kwargs", zipped_fmri)
def test_pairwiseannotations_fmri(DF, dims, annot_kwargs):
    DA = plst.DataAnalysis(
        data=DF, dims=dims, verbose=True, subject="subject"
    )  # !! subject
    _ph = DA.test_pairwise(paired=True, padjust="bonf")
    DA = (
        DA.subplots()
        .fillaxes(kind="box")
        .annotate_pairwise(
            **annot_kwargs,
            show_ph=False,
            only_sig="strict",
        )
    )
    ### Don't plot while executing pytest in terminal
    if __name__ != "__main__":
        plt.close()


# %% For dataset qPCR


QPCR_annot_pairwise_kwargs = [
    dict(
        include=["F1", "LOXL2", "SOST"],
        exclude=["F2", {"MMP7": ("F1", "F3")}],
        include_in_facet={
            "MMPs": ["MMP7", {"MMP9": ("F1", "F2")}],
            "Bone Metabolism": ["SOST", "F2", {"TIMP1": ("F3", "F1")}],
        },
        exclude_in_facet={
            "Wash": ["MMP7", {"MMP9": ("F1", "F2")}],
            "MACS": ["SOST", {"JAK2": ("F1", "F2")}],
        },
    ),
    dict(
        include=["F1", "LOXL2", "SOST"],
        exclude=["F2", {"MMP7": ("F1", "F3")}],
        include_in_facet={
            "MMPs": ["MMP7", {"MMP9": ("F1", "F2")}],
            "Bone Metabolism": ["SOST", "F2", {"TIMP1": ("F3", "F1")}],
        },
        exclude_in_facet={
            "Wash": ["MMP7", {"MMP9": ("F1", "F2")}],
            "MACS": ["SOST", {"JAK2": ("F1", "F2")}],
        },
    ),
    dict(
        include="__HUE",
        exclude=["F2", {"MMP7": ("F1", "F3")}],
    ),
    dict(
        include="__X",
        exclude=["F2", {"MMP7": ("F1", "F3")}],
    ),
    dict(
        include=["Vimentin", "MMP7"],
        exclude=["FZD4"],
    ),
]

zipped_qpcr: list[tuple] = dac.add_zip_column(
    dac.zipped_noempty_qpcr, QPCR_annot_pairwise_kwargs
)


@pytest.mark.parametrize("DF, dims, annot_kwargs", zipped_qpcr)
def test_pairwiseannotation_qpcr(DF, dims, annot_kwargs):
    DA = plst.DataAnalysis(data=DF, dims=dims, verbose=True)
    _ph = DA.test_pairwise(paired=False, padjust="none", subject="subject")
    DA = (
        DA.subplots(sharey=False, figsize=(10, 10))
        .fillaxes(kind="box")
        .transform_y("log10")  # !! log transform
        .edit_y_scale_log(10)  # !! MUST be called before annotation!
        .annotate_pairwise(
            # include="__HUE",
            show_ph=False,
            only_sig="tolerant",
            **annot_kwargs,
        )
        # .edit_tight_layout() # !! just uglier
    )
    ### Don't plot while executing pytest in terminal
    if __name__ != "__main__":
        plt.close()


### Run without pytest
if __name__ == "__main__":
    DF, dims = plst.load_dataset("qpcr")
    AN = Annotator(data=DF, dims=dims, verbose=True)
    AN.levels_dendrogram()
    test_pairwiseannotation_qpcr(
        DF, dims, annot_kwargs=QPCR_annot_pairwise_kwargs[0]
    )

# %% Interactive testing to display Plots

if __name__ == "__main__":
    import ipytest

    ipytest.run()
