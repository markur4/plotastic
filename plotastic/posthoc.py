#
# %% Import Modules
import markurutils as ut

import pandas as pd
import pingouin as pg

# from plotastic.analysis import Analysis
from plotastic.assumptions import Assumptions


# %% Class PostHoc


class PostHoc(Assumptions):
    STANDARD_KWS = dict(
        nan_policy="pairwise",  # * Delete only pairs or complete subjects ("listwise") if sasmples are missing?
        return_desc=True,  # * Return descriptive statistics?
        correction="auto",  # * Use welch correction if variances unequal?
    )

    # ... __INIT__ .......................................................#
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)

    # ... Base functions ..................................................#

    @ut.ignore_warnings
    def _base_pairwise_tests(self, **kwargs) -> pd.DataFrame:
        """Performs pairwise tests for a facet of self.data"""

        ### Turn around hue and x for between or within argument
        if self.dims.hue:
            kwargs_2 = kwargs.copy()
            if "within" in kwargs:
                kwargs_2["within"] = list(reversed(kwargs["within"]))
            elif "between" in kwargs:
                kwargs_2["between"] = list(reversed(kwargs["between"]))

        ### Iterate over rows and columns
        PH_dict = {}
        for key, df in self.data_iter__key_rowcol:
            ### Perform pairwise tests
            if self.dims.hue:  # * perform twice with x and hue turned around (= huex)
                ph_xhue = pg.pairwise_tests(data=df, **kwargs)
                ph_huex = pg.pairwise_tests(data=df, **kwargs_2)
                PH_dict[key] = ph_xhue.merge(ph_huex, how="outer")
            else:  # * perform once with x
                ph_x = pg.pairwise_tests(data=df, **kwargs)
                PH_dict[key] = ph_x

        PH = pd.concat(PH_dict, keys=PH_dict.keys(), names=self.factors_rowcol_list)
        return PH

    def _enhance_PH(self, PH: pd.DataFrame) -> pd.DataFrame:
        return PH

    # ... T TEST ..................................................#

    def test_pairwise(
        self, paired=True, parametric=True, subject=None, **user_kwargs
    ) -> pd.DataFrame:
        """Multiple paired tests"""

        ### Gather Arguments
        # *
        kwargs = dict(
            dv=self.dims.y,
            parametric=parametric,
            nan_policy="pairwise",
        )

        # * Paired or unpaired
        if paired:
            assert (self.subject is not None) or (
                subject is not None
            ), "No subject column specified"
            kwargs["within"] = list(self.factors_xhue)
            kwargs["subject"] = self.subject if self.subject else subject
        else:
            kwargs["between"] = list(self.factors_xhue)

        # * Add user kwargs
        kwargs.update(self.STANDARD_KWS)
        kwargs.update(user_kwargs)

        ### Make PH table
        PH = self._base_pairwise_tests(**kwargs)
        PH = self._enhance_PH(PH)

        return PH


# %% Get Data and make DataAnalysis object

DF, dims = ut.load_dataset("fmri")
DA = PostHoc(data=DF, dims=dims, subject="subject")


# %%
DA.test_pairwise(paired=False, padjust="bonf", return_desc=False)

# %%
DA.test_multiple_t(padjust="bonf")

# %%

pg.pairwise_tests(
    data=DF,
    dv="signal",
    # between=[dims["x"], dims["hue"]],
    within=[dims["x"], dims["hue"]],
    between=dims["col"],
    subject="subject",
    parametric=True,
    padjust="bh",
    nan_policy="pairwise",
)

# %%
