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
        for (
            key,
            df,
        ) in (
            self.data_iter__key_facet_skip_empty
        ):  # * no empty means that empty groups are skipped
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
        """Interface that sorts arguments, executes pairwise tests and adds extra features to PH table"""

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

        # * Make sure the specified factors are present
        if "within" in kwargs:
            assert all(
                [f in self.factors_xhue for f in kwargs["within"]]
            ), f"Argument 'within' contains unknown columns ({kwargs['within']} should be like one of {self.factors_all}"
        if "between" in kwargs:
            assert all(
                [f in self.factors_xhue for f in kwargs["between"]]
            ), f"Argument 'between' contains unknown columns ({kwargs['between']} should be like one of {self.factors_all}"

        ### Make PH table
        PH = self._base_pairwise_tests(**kwargs)
        PH = self._enhance_PH(PH)

        return PH


# %% Import data and make PostHoc object


DF, dims = ut.load_dataset("tips")

PH = PostHoc(data=DF, dims=dims, verbose=False)


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
