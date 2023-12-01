#
# %% Import Modules
# import markurutils as ut
import plotastic.utils.utils as ut

import numpy as np
import pandas as pd

# print(pd.__version__)
import pingouin as pg


from plotastic.stat.assumptions import Assumptions


# %% Class PostHoc


class PostHoc(Assumptions):
    DEFAULT_KWS_PAIRWISETESTS = dict(
        nan_policy="pairwise",  #' Delete only pairs or complete subjects ("listwise") if sasmples are missing?
        return_desc=True,  #' Return descriptive statistics?
        correction="auto",  #' Use welch correction if variances unequal?
    )

    # == __init__ ======================================================================
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    #
    #
    # == Base function =================================================================

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
        #' Iterate over rows and columns
        PH_dict = {}

        #' Skip empty so that no empty groups of level combinations are artificially added
        for key, df in self.data_iter__key_facet_skip_empty:
            # print(key)
            # ut.pp(df)

            # for key in self.levelkeys_rowcol:
            #     df = self.data_dict_skip_empty[key]

            if (
                self.dims.hue
            ):  #' Perform twice with x and hue turned around (= huex)
                ph_xhue = pg.pairwise_tests(data=df, **kwargs)
                ph_huex = pg.pairwise_tests(data=df, **kwargs_2)
                PH_dict[key] = ph_xhue.merge(ph_huex, how="outer")
            else:  #' perform once with x
                ph_x = pg.pairwise_tests(data=df, **kwargs)
                PH_dict[key] = ph_x

        PH = pd.concat(
            PH_dict, keys=PH_dict.keys(), names=self.factors_rowcol_list
        )

        return PH

    #
    # == Pairwise TESTs ================================================================

    def test_pairwise(
        self,
        paired=None,
        parametric=True,
        subject=None,
        # only_contrast=False,
        **user_kwargs,
    ) -> pd.DataFrame:
        """Interface that sorts arguments, executes pairwise tests and adds extra features to PH table"""

        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            parametric=parametric,
            nan_policy="pairwise",
        )
        #' Paired or unpaired
        if paired is None and self.subject:
            paired = True
        if paired:
            assert (self.subject is not None) or (
                subject is not None
            ), "No subject column specified"
            kwargs["within"] = self.factors_xhue
            kwargs["subject"] = self.subject if self.subject else subject
        else:
            kwargs["between"] = self.factors_xhue
        #' Add user kwargs
        kwargs.update(self.DEFAULT_KWS_PAIRWISETESTS)
        kwargs.update(user_kwargs)

        #' Make sure the specified factors are present
        if "within" in kwargs:
            assert all(
                [
                    f in self.factors_all
                    for f in ut.ensure_list(kwargs["within"])
                ]
            ), f"Argument 'within' contains unknown columns ({kwargs['within']} should be like one of {self.factors_all}"
        if "between" in kwargs:
            assert all(
                [
                    f in self.factors_all
                    for f in ut.ensure_list(kwargs["between"])
                ]
            ), f"Argument 'between' contains unknown columns {kwargs['between']} should be like one of {self.factors_all}"

        ### Make PH table
        PH = self._base_pairwise_tests(**kwargs)
        PH = self._enhance_PH(
            PH,
            # only_contrast=only_contrast,
        )

        ### Save result
        self.results.DF_posthoc = PH

        return PH

    def _enhance_PH(
        self,
        PH: pd.DataFrame,
        # only_contrast=False,
    ) -> pd.DataFrame:
        ### Define Alpha
        alpha = self.ALPHA
        alpha_tolerance = self.ALPHA_TOLERANCE

        ### Define column that contains p-values
        # pcol = "p-unc" if padjust in ("none", None) else "p-corr"

        ### EDIT PH
        PH = PH.reset_index(
            drop=False
        )  #' drop is default false, but put it explicitly  here

        #' Add Stars
        PH["**p-unc"] = PH["p-unc"].apply(self._p_to_stars, alpha=alpha)
        if "p-corr" in PH.columns:
            PH["**p-corr"] = PH["p-corr"].apply(self._p_to_stars, alpha=alpha)

        #' Make Column for categorizing significance
        PH["Sign."] = pd.cut(
            PH["p-unc"],
            bins=[0, alpha, alpha_tolerance, 1],
            labels=["signif.", "toler.", False],
        )
        if "p-corr" in PH.columns:
            PH["Sign."] = pd.cut(
                PH["p-corr"],
                bins=[0, alpha, alpha_tolerance, 1],
                labels=["signif.", "toler.", False],
            )

        #' Make pairs
        PH["pairs"] = PH.apply(self._level_to_pair, axis=1)

        # ### Check contrast
        # #' Optionally remove non-contrast comparisons
        # if only_contrast and self.dims.hue:
        #     PH = PH[
        #         PH["Contrast"].str.contains("*", regex=False)
        #     ]  # <<<< OVERRRIDE PH, REMOVE NON-CONTRAST ROWS

        #' Show if the pair crosses x or hue boundaries
        if self.dims.hue:
            PH["cross"] = PH.apply(self._detect_xhue_crossing, axis=1)
        else:
            PH["cross"] = "x"

        ### Set index
        PH = ut.drop_columns_by_regex(PH, "level_\d")
        if self.dims.hue:
            PH = PH.set_index(
                self.factors_rowcol_list + [self.dims.hue, "Contrast"]
            )
        else:
            PH = PH.set_index(self.factors_rowcol_list + ["Contrast"])

        return PH

    # == Pairing functions =============================================================

    def _level_to_pair(self, row: "pd.Series") -> tuple:
        """converts the factor-columns of a posthoc table into a column of pairs"""

        ### See if there are multiple factors
        phInteract = " * " in row["Contrast"]

        if not phInteract:
            return row["A"], row["B"]
        else:
            levels = row[[self.dims.hue, self.dims.x]].tolist()
            if pd.notna(
                levels[0]
            ):  # switch column if NaN, also check: if not math.isnan(factor)
                lvl = levels[0]
                pair = ((row["B"], lvl), (row["A"], lvl))
            else:
                lvl = levels[1]
                pair = ((lvl, row["B"]), (lvl, row["A"]))
        return pair

    @staticmethod
    def _detect_xhue_crossing(row: "pd.Series") -> str:
        """
        Detects if a pair ((DCN, F2), (DCN, F1)) is crossing x or hue boundaries
        :param row:
        :return:
        """

        """crossing Hue: ((x, hue1), (x, hue2))"""
        """crossing X:   ((x1, hue), (x2, hue))"""

        ### See if there are multiple factors
        phInteract = " * " in row["Contrast"]

        if not phInteract:
            return "x"
        else:
            cross = np.nan
            pair = row["pairs"]
            if pair[0][0] == pair[1][0]:
                cross = "hue"
            if pair[0][1] == pair[1][1]:
                cross = "x"
            return cross


# %% Import data and make PostHoc object


# DF, dims = plst.load_dataset("fmri")

# PH = PostHoc(data=DF, dims=dims, verbose=False, subject="subject")


# %% Check functionality of pingouin

# # !! Raises TypeError: Could not convert value 'cuestim' to numeric. This didn't happen before changing to new environment.
# # !! Downgraded pandas from 2.0.3 (released april 2023) to 1.5.3 -> FIXED IT
# ph = pg.pairwise_tests(data=DF, dv="signal", within=["timepoint", "event"], subject="subject", parametric=True, padjust="bonf", nan_policy="pairwise")

# %% test with pingouin

# ph = PH.test_pairwise(
#     # dv="signal",
#     # between=[dims["x"], dims["hue"]],
#     # within=[dims["x"], dims["hue"]],
#     # between=dims["col"],
#     # subject="subject",
#     parametric=True,
#     padjust="bh",
#     nan_policy="pairwise",
# )

# ut.pp(ph[ph["Sign."].isin(["signif."])]).head(70)

# %%
