#
# %%

import pandas as pd
import pingouin as pg

from plotastic.dimensions.dataframetool import DataFrameTool
from plotastic.stat.statresults import StatResults
from plotastic.stat.stattest import StatTest


# %% Class Assumptions


class Assumptions(StatTest):
    # == __init__=======================================================================
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    #
    #
    # == Normality =====================================================================

    def check_normality(self, method: str = "shapiro", **user_kwargs) -> pd.DataFrame:
        """_summary_

        :param method: 'shapiro', 'jarque-bera' or 'normaltest', defaults to 'shapiro'
        :type method: str, optional
        :return: _description_
        :rtype: pd.DataFrame
        """

        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            group=self.dims.x,  # ! pingouin crashes without group, so we iterate without x
            method=method,
        )
        kwargs.update(user_kwargs)  # * Add user kwargs

        ### Perform Test
        # * Iterate over rows, cols, hue
        normDF_dict = {}

        # * Skip empty groups
        for key, df in self.data_iter__key_groups_skip_empty:
            # * key = (row, col, hue)
            normdf = pg.normality(df, **kwargs)
            # * Add n to seehow big group is
            normdf["n"] = df.groupby(self.dims.x).count()[self.dims.y]

            normDF_dict[key] = normdf
        normDF = pd.concat(normDF_dict, keys=normDF_dict.keys(), names=self.factors_all)

        ### Save Results
        self.results.DF_normality = normDF

        return normDF

    #
    # == Homoscedasticity ==============================================================

    def check_homoscedasticity(
        self, method: str = "levene", **user_kwargs
    ) -> pd.DataFrame:
        """_summary_

        :param method: 'levene' or 'bartlett', defaults to "levene"
        :type method: str, optional
        :return: _description_
        :rtype: pd.DataFrame
        """

        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            group=self.dims.x,  # ! required, homoscedasticity is measured over a list of groups
            method=method,
        )
        kwargs.update(user_kwargs)  # * Add user kwargs

        ### Perform Test
        # * Iterate over rows, cols, and hue
        homoscedDF_dict = {}

        # * Skip empty groups
        for key, df in self.data_iter__key_groups_skip_empty:
            # * key = (row, col, hue)
            homosceddf = pg.homoscedasticity(df, **kwargs)
            # * Add number of groups
            homosceddf["group count"] = df.groupby(self.dims.x).count().shape[0]
            # * Add n to seehow big groups are
            homosceddf["n per group"] = [df.groupby(self.dims.x).count()[self.dims.y].to_list()]

            homoscedDF_dict[key] = homosceddf
        homoscedDF = pd.concat(
            homoscedDF_dict, keys=homoscedDF_dict.keys(), names=self.factors_all
        )

        ### Save Results
        self.results.DF_homoscedasticity = homoscedDF

        return homoscedDF


# ! end class
# !
# !


# %% Import Data

import markurutils as ut

DF, dims = ut.load_dataset("fmri")

# %% plot
import seaborn as sns

sns.catplot(data=DF, **dims, kind="box")


# %% Check functionality with pingouin

pg.normality(DF, dv=dims["y"], group=dims["x"])
pg.homoscedasticity(DF, dv=dims["y"], group=dims["x"])

# %% create Assumptions object

DA = Assumptions(data=DF, dims=dims)

# %% Check normality

DA.check_normality()

# %% Check homoscedasticity

DA.check_homoscedasticity()
