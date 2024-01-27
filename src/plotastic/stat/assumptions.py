#
# %%

from typing import TYPE_CHECKING, NamedTuple  #' SpherResults is a NamedTuple

import pandas as pd
import pingouin as pg

# from plotastic.dimensions.dataframetool import DataFrameTool
# from plotastic.stat.statresults import StatResults
from plotastic.stat.stattest import StatTest

# if TYPE_CHECKING:
#     from collections import namedtuple #' SpherResults is a NamedTuple
#     # from pingouin.distribution import SpherResults

# %% Class Assumptions


class Assumptions(StatTest):
    # == __init__=======================================================================
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

        self.two_factor = True

    # ==
    # ==
    # == Normality =====================================================================

    def check_normality(
        self, method: str = "shapiro", **user_kwargs
    ) -> pd.DataFrame:
        """Check assumption of normality. If the assumption is violated, you should use
        non-parametric tests (e.g. Kruskal-Wallis, Mann-Whitney, Wilcoxon, etc.) instead
        of parametric tests (ANOVA, t-test, etc.).


        :param method: 'shapiro', 'jarque-bera' or 'normaltest', defaults to 'shapiro'
        :type method: str, optional
        :return: _description_
        :rtype: pd.DataFrame
        """

        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            group=self.dims.x,  # !! pingouin crashes without group, so we iterate without x
            method=method,
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform Test
        #' Iterate over rows, cols, hue
        #' Skip empty groups
        normDF_dict = {}
        # TODO: Use an iterator from hierarchical instead of one that omits x
        for key, df in self.data_iter__key_groups_skip_empty:
            #' key = (row, col, hue)
            normdf = pg.normality(df, **kwargs)
            #' Add n to seehow big group is.
            normdf["n"] = self.data_count_n_per_x(
                df
            )  #' -> Series with same length as normdf

            normDF_dict[key] = normdf

        normDF = pd.concat(
            normDF_dict, keys=normDF_dict.keys(), names=self.factors_all
        )

        ### Save Results
        self.results.DF_normality = normDF

        return normDF

    # ==
    # == Homoscedasticity ==============================================================

    def check_homoscedasticity(
        self, method: str = "levene", **user_kwargs
    ) -> pd.DataFrame:
        """Checks assumption of homoscedasticity. If the assumption is violated, the
        p-values from a t-test should be corrected with Welch's correction.

        :param method: 'levene' or 'bartlett', defaults to "levene"
        :type method: str, optional
        :return: _description_
        :rtype: pd.DataFrame
        """

        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            group=self.dims.x,  # !! required, homoscedasticity is measured over a list of groups
            method=method,
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform Test
        #' Iterate over rows, cols, and hue
        #' Skip empty groups
        homosced_dict = {}
        for key, df in self.data_iter__key_groups_skip_empty:
            #' key = (row, col, hue)
            homosced = pg.homoscedasticity(df, **kwargs)
            #' Add number of groups
            homosced["group count"] = self.data_count_groups_in_x(df)
            #' Add n to see how big groups are, make nested list to fit into single cell
            homosced["n per group"] = [self.data_count_n_per_x(df).to_list()]

            homosced_dict[key] = homosced

        homoscedDF = pd.concat(
            homosced_dict, keys=homosced_dict.keys(), names=self.factors_all
        )

        #
        ### Save Results
        self.results.DF_homoscedasticity = homoscedDF

        return homoscedDF

    # ==
    # == Spherecity ====================================================================

    @staticmethod
    def _spher_to_df(spher: NamedTuple) -> pd.DataFrame:
        """pingouin returns a strange SpherResults object (namedtuple?), we need to
        convert it to a dataframe.

        :param spher: Output of pg.sphericity()
        :type spher: pingouin.distribution.SpherResults, NamedTuple
        :return: Sphericity Result as DataFrame
        :rtype: pd.DataFrame
        """

        if isinstance(spher, tuple):
            spher_dict = dict(zip(["spher", "W", "chi2", "dof", "pval"], spher))
            spher_DF = pd.DataFrame(data=spher_dict, index=[0])
        else:
            spher_DF = pd.DataFrame(data=spher._asdict(), index=[0])

        return spher_DF

    def check_sphericity(
        self, method: str = "mauchly", **user_kwargs
    ) -> pd.DataFrame:
        """Checks assumption of sphericity. If the assumption is violated, the p-values
        of an RM-ANOVA should be corrected with Greenhouse-Geisser or Huynh-Feldt method

        :param method: 'mauchly' or 'jns', defaults to "mauchly"
        :type method: str, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        ### Make sure subject is specified
        if self.subject is None:
            raise ValueError(
                "Testing sphericity requires a subject to be specified."
            )
        
        # TODO: Add option to use x or hue as within-factors
        ### All

        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            subject=self.subject,
            within=self.dims.x,
            method=method,
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform Test
        #' Iterate over rows, cols, and hue
        #' Skip empty groups
        spher_dict = {}
        for key, df in self.data_iter__key_groups_skip_empty:
            #' key = (row, col, hue)
            spher = pg.sphericity(df, **kwargs)
            #' Convert NamedTuple to DataFrame
            spherdf = self._spher_to_df(spher)
            #' Add number of groups
            spherdf["group count"] = self.data_count_groups_in_x(df)
            #' Add n to seehow big groups are
            spherdf["n per group"] = [self.data_count_n_per_x(df).to_list()]

            spher_dict[key] = spherdf

        spherDF = pd.concat(
            spher_dict, keys=spher_dict.keys(), names=self.factors_all_without_x
        )

        ### Save Results
        self.results.DF_sphericity = spherDF

        return spherDF


# !! end class
# !!
# !!


# #%%
# from plotastic.example_data.load_dataset import load_dataset
# DF, dims = load_dataset("fmri")


# # %% plot
# import seaborn as sns

# sns.catplot(data=DF, **dims, kind="box")

# # %% Check functionality with pingouin

# pg.normality(DF, dv=dims["y"], group=dims["x"])
# pg.homoscedasticity(DF, dv=dims["y"], group=dims["x"])

# spher = pg.sphericity(DF, dv=dims["y"], subject="subject", within=dims["x"])
# type(spher)

# # %% create Assumptions object

# DA = Assumptions(data=DF, dims=dims, subject="subject", verbose=True)

# DA.check_normality()
# DA.check_homoscedasticity()
# DA.check_sphericity()

# #%% Plot roughest facetting

# sns.catplot(data=DF, x="timepoint")

# # %% Use different set


# DA2 = Assumptions(data=DF, dims=dict(x="timepoint", y="signal"),
#                   subject="subject", verbose=True)
# # DA2.catplot()

# DA2.check_normality()
# DA2.check_homoscedasticity()
# DA2.check_sphericity()
