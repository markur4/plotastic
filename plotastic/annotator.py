#
# %%
import markurutils as ut

from plotastic.posthoc import PostHoc
from plotastic.multiplot import MultiPlot
from plotastic.omnibus import Omnibus
from plotastic.bivariate import Bivariate

# %% class Annotator


class Annotator(MultiPlot, Omnibus, PostHoc, Bivariate):
    # ... ­­init__ .....................................................................

    def __init__(self, **dataframetool_kws):
        ### Inherit
        # * verbosity false, since each subclass can test its own DataFrame
        super().__init__(**dataframetool_kws)

    # ... ANNOTATE POSTHOC  :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ... PostHoc: Main Annotation Function ...............................................................

    def annotate_pairs(self):
        """Annotate pairs of groups with pairwise tests."""

        ### Assert presence of a posthoc table
        # * We could execute automatically, but producing a plot and a posthoc test at the same time is a lot to handle
        assert (
            not self.results.DF_posthoc is "NOT TESTED"
        ), "Posthoc not tested yet, please call .test_pairwise() first"

        ###
        return self

    # ... PostHoc: Check Pairs .......................................................

    def blah(self):
        pass


# %%

DF, dims = ut.load_dataset("fmri")
AN = Annotator(
    data=DF,
    dims=dims,
    subject="subject",
    verbose=True,
)

AN.test_pairwise()

# %%

AN = AN.subplots().fillaxes(kind="box").annotate_pairs()
# %%
