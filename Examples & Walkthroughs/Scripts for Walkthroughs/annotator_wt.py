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

