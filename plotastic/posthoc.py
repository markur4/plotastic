#
# %% Import Modules
import markurutils as ut

import pandas as pd
import pingouin as pg


from plotastic.assumptions import Assumptions


# %% Class PostHoc


class PostHoc(Assumptions):
    STANDARD_KWS = dict(
        nan_policy="pairwise",  # * Delete only pairs or complete subjects ("listwise") if sasmples are missing?
        return_desc=True,  # * Return descriptive statistics?
        correction="auto",  # * Use welch correction if variances unequal?
    )

    # ... __INIT__ .......................................................#
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

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

        ### Perform Test
        # * Iterate over rows and columns
        PH_dict = {}
        for key, df in self.data_iter__key_facet_skip_empty: #* no empty means that empty groups are skipped
            # print(key)
            # ut.pp(df)

            if self.dims.hue:  # * perform twice with x and hue turned around (= huex)
                ph_xhue = pg.pairwise_tests(data=df, **kwargs)
                ph_huex = pg.pairwise_tests(data=df, **kwargs_2)
                PH_dict[key] = ph_xhue.merge(ph_huex, how="outer")
            else:  # * perform once with x
                ph_x = pg.pairwise_tests(data=df, **kwargs)
                PH_dict[key] = ph_x

        return pd.concat(PH_dict, keys=PH_dict.keys(), names=self.factors_rowcol_list)

    def _enhance_PH(self, PH: pd.DataFrame) -> pd.DataFrame:
        return PH

    # ... Pairwise TESTs ..................................................#

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
            kwargs["within"] = self.factors_xhue
            kwargs["subject"] = self.subject if self.subject else subject
        else:
            kwargs["between"] = self.factors_xhue

        # * Add user kwargs
        kwargs.update(self.STANDARD_KWS)
        kwargs.update(user_kwargs)

        ### Make PH table
        PH = self._base_pairwise_tests(**kwargs)
        PH = self._enhance_PH(PH)

        return PH


# %% make iterator
from plotastic.dataframetool import DataFrameTool

DF, dims = ut.load_dataset("tips")

DA = DataFrameTool(data=DF, dims=dict(y="tip", x="size-cut"), verbose=False)
# DA = PostHoc(data=DF, dims=dict(y="tip", x="size-cut"), verbose=True)


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


# # %% test for dataset fmri

# DF, dims = ut.load_dataset("fmri")
# DA = PostHoc(
#     data=DF,
#     dims=dims,
#     subject="subject",
#     verbose=True,
# )
# # DA.data_describe()

# DA.test_pairwise(
#     paired=True,
#     padjust="bonf",
#     subject="subject",
#     return_desc=False,
# )


# # %%  test for dataset tips

# DF, dims = ut.load_dataset("tips")
# DA = PostHoc(
#     data=DF,
#     dims=dims,
#     subject="day",
#     verbose=True,
# )
# # DA.data_describe()

# DA.test_pairwise(
#     paired=False,  # ! If using paired, but table is not paired, it will throw a cryptic error "AssertionError: The T-value must be a int or a float."
#     padjust="bonf",
#     # subject="subject",
#     return_desc=False,
# )


# %% test with pingouin

# pg.pairwise_tests(
#     data=DF,
#     dv="signal",
#     # between=[dims["x"], dims["hue"]],
#     within=[dims["x"], dims["hue"]],
#     between=dims["col"],
#     subject="subject",
#     parametric=True,
#     padjust="bh",
#     nan_policy="pairwise",
# )

# %%
