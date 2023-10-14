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
            group=self.dims.x,
        )
        # * Add user kwargs
        kwargs.update(user_kwargs)

        ### Perform Test
        # * Iterate over rows and cols
        normDF_dict = {}

        # * Skip empty groups
        for key, df in self.data_iter__key_facet_skip_empty:
            normdf =  pg.normality(df, **kwargs)
            normdf["n"] = df.count()[self.dims.y] #* Add n to seehow big group is 
            
            normDF_dict[key] = normdf
            

        normDF = pd.concat(
            normDF_dict, keys=normDF_dict.keys(), names=self.factors_rowcol_list
        )

        return normDF


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

pg.normality(DF, dv=dims["y"], group= dims["x"])

# %% create Assumptions object

DA = Assumptions(data=DF, dims=dims)

# %% Check normality

DA.check_normality()

